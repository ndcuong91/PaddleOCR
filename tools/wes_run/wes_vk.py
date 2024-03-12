
import pandas as pd
import os

def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG', 'jpeg', 'JPEG']):
    included_extensions = ext
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    file_names = sorted(file_names)
    return file_names

def find_duplicate(df1, df2):
    common = df1.merge(df2, how = 'inner' ,indicator=False)
    return common

def find_duplicate_in_dir(src_dir):
    list_files = get_list_file_in_folder(src_dir, ext=['xlsx'])
    df1 = pd.read_excel(os.path.join(src_dir, list_files[0]))
    for idx, file in enumerate(list_files):
        print(idx, file)
        df2 = pd.read_excel(os.path.join(src_dir, file))
        df1 = find_duplicate(df1, df2)
    df1.to_excel("PT_duplicate_final.xlsx")



def get_raw_data_from_combine(combine_path, ori_path):

    df1 = pd.read_excel(combine_path)
    df2 = pd.read_excel(ori_path)   #Chr	Start	End	Ref	Alt	Func.refGene	Gene.refGene	avsnp150

    final = pd.DataFrame()
    for index, row in df1.iterrows():
        print(index, row)
        ori_row = df2.loc[(df2['Chr'] == row['Chr']) &
                          (df2['Start'] == row['Start']) &
                          (df2['End'] == row['End']) &
                          (df2['Ref'] == row['Ref']) &
                          (df2['Alt'] == row['Alt']) &
                          (df2['Func.refGene'] == row['Func.refGene']) &
                          (df2['Gene.refGene'] == row['Gene.refGene']) &
                          (df2['avsnp150'] == row['avsnp150'])]
        final = pd.concat([final, ori_row], ignore_index=True)
    final.to_excel('final.xlsx')

if __name__=='__main__':
    # df1 = pd.read_excel('/home/misa/PycharmProjects/PaddleOCR/tools/wes_run/super_short2/PT10 EXCEL.xlsx')
    # df2 = pd.read_excel('/home/misa/PycharmProjects/PaddleOCR/tools/wes_run/super_short2/PT11 EXCEL.xlsx')
    # find_duplicate(df1,df2)
    # find_duplicate_in_dir('/home/misa/PycharmProjects/PaddleOCR/tools/wes_run')
    # df1 = pd.read_excel('/home/misa/PycharmProjects/PaddleOCR/tools/wes_run/PT_duplicate.xlsx')
    # df1 = df1.drop_duplicates()
    # df1.to_excel("PT_duplicate_fina2l.xlsx")

    get_raw_data_from_combine('/home/misa/PycharmProjects/PaddleOCR/tools/wes_run/combine.xlsx','/home/misa/PycharmProjects/PaddleOCR/tools/wes_run/final3.xlsx')

