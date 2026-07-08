import pandas as pd
import numpy as np
import re
from pathlib import Path
pd.set_option('future.no_silent_downcasting', True)

def header_merge(df):
    """
    병합되어 사용되던 1~3행의 컬럼을 1행으로 합칩니다.
    """
    # 헤더 정보가 있는 상위 2개 행에 대해 forward fill 적용
    df.iloc[0:2, :] = df.iloc[0:2, :].ffill(axis=1)
    new_columns = df.iloc[0:3, :].fillna('').apply(lambda x: '_'.join(x.astype(str)).strip().rstrip('_'))
    df.columns = new_columns
    
    # 헤더로 사용된 기존 0, 1, 2번 행 삭제
    df = df.drop(index=[0, 1, 2])
    return df

def drop_column(df):
    """
    불필요한 집계 데이터를 제거합니다.
    """
    # 집계 열 제거
    if df.columns.str.contains('증감|평균|대비|수능최저').any():
        df = df.loc[:, ~df.columns.str.contains('증감|평균|대비|수능최저')]

    # 집계 행 제거
    df = df.dropna(subset=[df.columns[1]])
    df = df.reset_index(drop=True)

    return df

def transform_to_tidy(df):
    """
    Wide 형태의 데이터를 Tidy 형태로 변환합니다.
    """
    df.columns = df.columns.str.replace(r'실질경쟁률_(\d{4})학년도_(.*)', r'\2_\1', regex=True) \
                           .str.replace(r'입결\((\d{4}).*?\)_어디가_(.*)', r'입결\2_\1', regex=True) \
                           .str.replace('학년도', '', regex=False).str.replace('\n', '', regex=False)

    id_vars = ['세부', '대학', '전형', '학과']
    df_long = df.melt(id_vars=id_vars, var_name='속성', value_name='값')
    
    extracted_data = df_long['속성'].str.extract(r'^(.*?)_?(\d{4})?$')
    df_long[['측정항목', '년도']] = extracted_data
    df_long['측정항목'] = df_long['측정항목'].fillna(df_long['속성'])
    
    df_tidy = df_long.pivot_table(
        index=id_vars + ['년도'],
        columns='측정항목',
        values='값',
        aggfunc='first'
    ).reset_index()

    df_tidy.columns.name = None

    df_tidy = df_tidy.replace(r'^\s*-\s*$', np.nan, regex=True)
    return df_tidy

def parse_stage_details(stage_string):
    """
    '전형방법' 문자열을 세부 요소(방법, 비중)로 분해하는 헬퍼 함수.
    """
    if pd.isna(stage_string):
        return pd.Series({'방법1': None, '비중1': None, '방법2': None, '비중2': None})
    
    components = stage_string.split('+')
    parsed_data = {}
    for i, comp in enumerate(components, 1):
        if i > 2: continue
        match = re.search(r'([^\d\s]+)\s*(\d+)', comp)
        if match:
            parsed_data[f'방법{i}'] = match.group(1).strip()
            parsed_data[f'비중{i}'] = int(match.group(2).strip())
        else:
            parsed_data[f'방법{i}'] = comp.strip()
            parsed_data[f'비중{i}'] = None
    return pd.Series(parsed_data)

def structure_admission_methods(df):
    """
    '전형방법' 열을 세부적인 구조화된 열로 변환합니다.
    """
    if '전형방법' not in df.columns:
        return df

    df['전형방법'] = df['전형방법'].fillna('')
    df['1단계'] = df['전형방법'].str.extract(r'\[1\](.*?)(?:\n|$)')[0].str.strip()
    df['2단계'] = df['전형방법'].str.extract(r'\[2\](.*?)(?:\n|$)')[0].str.strip()
    df['일괄전형'] = df['전형방법'].str.extract(r'\[일괄\](.*)')[0].str.strip()
    
    df['1단계_통합'] = df['1단계'].fillna(df['일괄전형'])
    
    stage1_details = df['1단계_통합'].apply(parse_stage_details).rename(columns={'방법1': '1-1차', '비중1': '1-1비중', '방법2': '1-2차', '비중2': '1-2비중'})
    stage2_details = df['2단계'].apply(parse_stage_details).rename(columns={'방법1': '2-1차', '비중1': '2-1비중', '방법2': '2-2차', '비중2': '2-2비중'})
    
    df_structured = pd.concat([df, stage1_details, stage2_details], axis=1)
    df_structured = df_structured.drop(columns=['전형방법', '1단계', '2단계', '일괄전형', '1단계_통합'])
    return df_structured

def update_combined_sheet(base_dir, output_csv_path):
    '''
    각 시트에 전처리를 진행하고 하나의 파일로 통합합니다.
    '''
    all_processed_data = []                           

    # 'num' 폴더와 그 하위 폴더에서 모든 엑셀 파일을 검색
    all_excel_files = list(base_dir.glob('**/*.xlsx')) + list(base_dir.glob('**/*.xls'))

    # 엑셀파일마다 작업 실행
    for excel_filepath in all_excel_files:
        print(f"\n{'='*40}\n📁 파일 처리 시작: {excel_filepath}\n{'='*40}")
        
        try:
            xls = pd.ExcelFile(excel_filepath)
            # 시트 이름에 '입결'이 포함된 이름만 모으기
            target_sheets = [name for name in xls.sheet_names if '입결' in name]
            
            if not target_sheets:
                print(f"'{excel_filepath}' 파일에 '입결' 시트가 없습니다. 건너뜁니다.")
                continue

            for sheet_name in target_sheets:          
                df_original = pd.read_excel(excel_filepath, sheet_name=sheet_name, header=None)
                df_prepared = header_merge(df_original)
                df_cleaned = drop_column(df_prepared)
                df_tidy = transform_to_tidy(df_cleaned)
                df_final = structure_admission_methods(df_tidy)

                all_processed_data.append(df_final)
                print(f"-> '{sheet_name}' 시트 처리 완료.")
                
        except Exception as e:
            print(f"'{excel_filepath}' 파일 처리 중 오류가 발생했습니다: {e}")

        # 각 시트 데이터를 하나의 df로 통합
        df_newly_processed = pd.concat(all_processed_data, ignore_index=True)
        print(f"총 {len(df_newly_processed)}개의 데이터 행을 처리했습니다.")

        # 통합 시트 생성 및 편집
        if not output_csv_path.exists():
            print(f"새로운 통합 파일 '{output_csv_path}'을 생성합니다.")
            df_newly_processed.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            final_rows = len(df_newly_processed)

        else:
            # 기존 파일의 핵심 칼럼만 수집
            unique_check_cols = ['세부', '대학', '전형', '학과', '년도']
            df_existing_keys = pd.read_csv(
                output_csv_path,
                usecols=unique_check_cols,
                dtype=str # 모든 키를 문자열로 읽어와서 타입 일치
            )

            # MultiIndex로 빠르게 중복여부 비교
            existing_multi_index = pd.MultiIndex.from_frame(df_existing_keys)
            is_duplicate = pd.MultiIndex.from_frame(df_newly_processed[unique_check_cols]).isin(existing_multi_index)

            # 중복되지 않은 행 추출
            df_to_append = df_newly_processed[~is_duplicate]

            if not df_to_append.empty:
                print(f"{len(df_to_append)}개의 새로운 데이터를 파일에 추가합니다.")
                df_to_append.to_csv(
                    output_csv_path,
                    mode='a',        # 'a'는 append 모드를 의미합니다.
                    header=False,    # 기존 파일이 있으므로 헤더는 추가하지 않습니다.
                    index=False,
                    encoding='utf-8-sig'
                )

            final_rows = len(df_existing_keys) + len(df_to_append)

            print(f"{len(df_to_append)}개의 데이터가 추가되어 총 {final_rows}개입니다.")



if __name__ == "__main__":
    # 현재 스크립트 파일(prep_car.py)의 절대 경로를 가져옵니다.
    current_file_path = Path(__file__).resolve()
    
    # 루트 디렉토리 경로를 설정합니다. (현재 파일의 부모 폴더('utils')의 부모 폴더)
    ROOT_DIR = current_file_path.parent.parent
 
    input_dir = ROOT_DIR / 'data' / 'car' # car폴더의 엑셀을 찾아서
    output_dir = ROOT_DIR / 'data'        # data폴더에 통합 데이터 만들기
    output_csv_path = output_dir / f"car_all_sheet.csv" # 전처리 완료 통합 데이터

    update_combined_sheet(input_dir, output_csv_path)
    
