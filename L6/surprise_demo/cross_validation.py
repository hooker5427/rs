from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import KFold

# cross_validate(算法，数据集，评估模块measures=[]，交叉验证折数cv)
# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# define a cross-validation iterator
kf = KFold(n_splits=3)

algo = SVD()
for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
