import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# 配置参数
input_files = ['../data/chemical_substructure_vector.csv',
               '../data/indication_vector.csv',
               '../data/target_vector.csv']
output_files = ['../data/chemical_substructure_vector_processed.csv',
                '../data/indication_vector_processed.csv',
                '../data/target_vector_processed.csv']

n_components_dict = {
    'data/chemical_substructure_vector': 40,
    'data/indication_vector.csv': 120,
    'data/target_vector.csv': 200
}


def remove_nan_columns(features):
    # 数据处理
    nan_mask = np.isnan(features).any(axis=0)
    valid_columns = ~nan_mask
    removed_cols = np.where(nan_mask)[0].tolist()
    cleaned_features = features[:, valid_columns]
    return cleaned_features, removed_cols


for input_file, output_file in zip(input_files, output_files):
    print(f"\n=== 正在处理文件: {input_file} ===")

    try:
        n_components = n_components_dict.get(input_file, 50)

        # 读取数据
        df = pd.read_csv(input_file, header=None)
        drug_ids = df.iloc[:, 0].astype(str)
        original_features = df.iloc[:, 1:].values

        # 清理数据
        features, removed_cols = remove_nan_columns(original_features)
        final_n_components = min(n_components, features.shape[1])
        if final_n_components < n_components:
            print(f"▏警告：目标维度自动调整为 {final_n_components}（原始维度不足）")

        #降维
        svd = TruncatedSVD(n_components=final_n_components)
        reduced_features = svd.fit_transform(features)

        #结果
        reduced_df = pd.DataFrame(
            reduced_features,
            columns=[f'PC{i + 1}' for i in range(final_n_components)]
        )
        reduced_df.insert(0, 'Drug_ID', drug_ids)

        #保存
        reduced_df.to_csv(output_file, index=False)
        cumulative_var = svd.explained_variance_ratio_.cumsum()[-1]
        print(f"▏最终维度: {final_n_components}")
        print(f"▏累计解释方差: {cumulative_var:.2%}")
        print(f"▏输出文件: {output_file}")

    except Exception as e:
        print(f"▏! 处理失败: {str(e)}")
        if 'features' in locals():
            print(f"▏当前有效维度: {features.shape[1]}")

print("\n=== 所有文件处理完成 ===")