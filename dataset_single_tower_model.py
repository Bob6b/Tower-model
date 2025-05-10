import pandas as pd
import numpy as np
import h5py  # 新增h5py库
from sklearn.model_selection import train_test_split

# 定义特征文件路径
FEATURE_FILES = {
    'feature1': '../data/chemical_substructure_vector_processed.csv',
    'feature2': '../data/indication_vector_processed.csv',
    'feature3': '../data/target_vector_processed.csv'
}


def load_feature_data():
    """加载所有特征数据"""
    features = {}
    for name, path in FEATURE_FILES.items():
        df = pd.read_csv(path, index_col=0)
        features[name] = df
    return features


def get_drug_features(drug_name, features):
    """获取单个药物的特征向量"""
    feature_vector = []
    for feature_name in ['feature1', 'feature2', 'feature3']:
        try:
            vec = features[feature_name].loc[drug_name].values.flatten().tolist()
            feature_vector.extend(vec)
        except KeyError:
            raise ValueError(f"药物{drug_name}在{feature_name}文件中不存在")
    return feature_vector


def process_dataset(sample_path, features, label):
    """处理正/负样本文件"""
    samples = []
    df = pd.read_csv(sample_path, header=None)

    for _, row in df.iterrows():
        drugA, drugB = row[0], row[1]

        try:
            vecA = get_drug_features(drugA, features)
            vecB = get_drug_features(drugB, features)
        except ValueError as e:
            print(f"跳过无效样本: {e}")
            continue

        combined = vecA + vecB + [label]
        samples.append(combined)

    return samples


def save_to_hdf5(train_df, test_df, filename="../dataset_single_tower_model/dataset.h5"):
    """将数据集存储为HDF5格式"""
    # 分离特征和标签
    X_train = train_df.drop('Label', axis=1).values.astype(np.float32)
    y_train = train_df['Label'].values.astype(np.int32)
    X_test = test_df.drop('Label', axis=1).values.astype(np.float32)
    y_test = test_df['Label'].values.astype(np.int32)

    # 创建HDF5文件
    with h5py.File(filename, 'w') as hf:
        # 创建训练组
        train_group = hf.create_group("train")
        train_group.create_dataset("features", data=X_train, compression="gzip", chunks=True)
        train_group.create_dataset("labels", data=y_train, compression="gzip", chunks=True)

        # 创建测试组
        test_group = hf.create_group("test")
        test_group.create_dataset("features", data=X_test, compression="gzip", chunks=True)
        test_group.create_dataset("labels", data=y_test, compression="gzip", chunks=True)

        # 添加元数据
        hf.attrs["creation_date"] = pd.Timestamp.now().isoformat()
        hf.attrs["description"] = "Drug interaction dataset_single_tower_model with chemical substructure, indication and target features"

        # 存储特征名称
        dt = h5py.special_dtype(vlen=str)
        feature_names = train_df.drop('Label', axis=1).columns.tolist()
        hf.create_dataset("feature_names", data=feature_names, dtype=dt)


# 加载特征数据
features = load_feature_data()

# 处理正负样本
pos_samples = process_dataset('twosides_interactions.csv', features, 1)
neg_samples = process_dataset('reliable_negatives.csv', features, 0)

# 合并数据集
all_samples = pos_samples + neg_samples
columns = [f'Feature_{i}' for i in range(len(all_samples[0]) - 1)] + ['Label']
dataset = pd.DataFrame(all_samples, columns=columns)

# 数据集分割
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

# 存储结果
train.to_csv('../dataset_single_tower_model/train_dataset.csv', index=False)
test.to_csv('../dataset_single_tower_model/test_dataset.csv', index=False)

# 新增HDF5存储功能
save_to_hdf5(train, test)

print("数据集构建完成，训练集样本数:", len(train), "测试集样本数:", len(test))
print("HDF5文件已生成：dataset.h5")