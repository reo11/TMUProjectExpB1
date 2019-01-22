# coding: utf-8
# 最終的なモデル詰め合わせ(スクリプトバージョン)

import numpy as np
import pandas as pd
from tqdm import tqdm

# kNN
from sklearn.neighbors import KNeighborsClassifier

# NeuralNet, CNN
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(0)

# LightGBM
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# 画像読み込み
from pathlib import Path
import cv2

# ニューラルネット用
class FlattenLayer(nn.Module):
    def forward(self, x):
        sizes = x.size()
        return x.view(sizes[0], -1)

class processes:
    def __init__(self, dbPath, queryPath, isFolder):
        if(isFolder):
            # 画像の場合targetの取得方法がないので強制的に、このcsvから取ってくる
            self.targetDBPath = "../input/Dlib/cutface/histFlattening/DB/csv/features_rel_dist.csv"
            self.targetQueryPath = "../input/Dlib/cutface/histFlattening/Query/csv/features_rel_dist.csv"
            self.db_df = pd.read_csv(self.targetDBPath)
            self.query_df = pd.read_csv(self.targetQueryPath)
            self.db_target_df = self.db_df["target"].copy()
            self.query_target_df = self.query_df["target"].copy()
            
            # DB画像読み込み
            p = Path(dbPath)
            p = sorted(p.glob("*.jpg"))
            self.dbImages = []
            for index, filename in enumerate(tqdm(p)):
                # 相対パスだと参照できなかったので絶対パスでやる
                img = cv2.imread(str(filename.resolve()), 0)
                # C, H, Wの形式にする(今回はグレースケールなのでC = 1)
                img = img.reshape([1, img.shape[0], img.shape[1]])
                self.dbImages.append((img/225).astype(np.float32))
                
            # Query画像読み込み
            p = Path(queryPath)
            p = sorted(p.glob("*.jpg"))
            self.queryImages = []
            for index, filename in enumerate(tqdm(p)):
                # 相対パスだと参照できなかったので絶対パスでやる
                img = cv2.imread(str(filename.resolve()), 0)
                # C, H, Wの形式にする(今回はグレースケールなのでC = 1)
                img = img.reshape([1, img.shape[0], img.shape[1]])
                self.queryImages.append((img/225).astype(np.float32))
            
        else:
            self.db_df = pd.read_csv(dbPath)
            self.query_df = pd.read_csv(queryPath)
            self.db_feature_df = self.db_df.drop(["target"], axis=1).copy()
            self.db_target_df = self.db_df["target"].copy()
            self.query_featture_df =self.query_df.drop(["target"], axis=1).copy()
            self.query_target_df = self.query_df["target"].copy()
            
            
    def calc_accuracy(self, process_type):
        # {"単純マッチング":0, "kNN":1, "NeuralNet":2, "LightGBM":3, "ピクセルマッチング":4, "CNN":5}
        if(process_type == 0):
            accuracy = self.simple_matching()
        elif(process_type == 1):
            accuracy = self.kNN()
        elif(process_type == 2):
            accuracy = self.NeuralNet()
        elif(process_type == 3):
            accuracy = self.lightGBM()
        elif(process_type == 4):
            accuracy = self.pixelMatching()
        elif(process_type == 5):
            accuracy = self.CNN()
        else:
            accuracy = -100
        return accuracy
            
            
    def simple_matching(self):
        prediction_df = pd.DataFrame(self.query_target_df)
        for i in range(self.query_featture_df.shape[0]):
            minimum_id = (((self.db_feature_df - self.query_featture_df.iloc[i])**2).sum(axis=1)).idxmin()
            prediction_df.loc[i, "predict"] = self.db_target_df[minimum_id]
            
        correct_num = (prediction_df["target"] == prediction_df["predict"]).sum()
        accuracy = correct_num / prediction_df.shape[0]
        return accuracy
    
    
    def kNN(self):
        DIV_NUM = 3 # k
        DIST_SETTING = 2 # ユークリッド=2, マンハッタン=1
        knn = KNeighborsClassifier(n_neighbors=DIV_NUM, p=DIST_SETTING, metric="minkowski")
        knn.fit(self.db_feature_df.values, self.db_target_df.values)
        prediction = knn.predict(self.query_featture_df.values)
        prediction_df = pd.DataFrame(self.query_target_df)
        prediction_df["predict"] = prediction
        correct_num = (prediction_df["target"] == prediction_df["predict"]).sum()
        accuracy = correct_num / prediction_df.shape[0]
        return accuracy
    
    def NeuralNet(self):
        input_size = self.db_feature_df.shape[1]
        # define network
        net = nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 20)
        )

        X_train = torch.tensor(self.db_feature_df.values, dtype=torch.float32)
        y_train = torch.tensor(self.db_target_df.values, dtype=torch.int64)
        
        X_test = torch.tensor(self.query_featture_df.values, dtype=torch.float32)
        y_test = torch.tensor(self.query_target_df.values, dtype=torch.int64)


        # 損失関数
        loss_fn = nn.CrossEntropyLoss()
        # adam
        optimizer = optim.Adam(net.parameters())
        # 損失ログ
        losses_train = []
        accuracy_test = []
        accuracy_train = []
        EPOCH = 300

        # 20エポック回す
        # ここだけ繰り返すと再学習しちゃうので注意
        for epoc in range(EPOCH):
            optimizer.zero_grad()
            
            y_pred = net(X_train)
            
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            
            optimizer.step()
            
            losses_train.append(loss.item())
            
            _, predicted = torch.max(y_pred, 1)
            corrects_train = 0
            for i in range(len(predicted)):
                if(predicted[i]==y_train[i]):
                    corrects_train += 1
            accuracy_train.append(corrects_train/len(y_train))
            
            y_test_pred = net(X_test)
            _, predicted = torch.max(y_test_pred, 1)
            corrects_test = 0
            for i in range(len(predicted)):
                if(predicted[i]==y_test[i]):
                    corrects_test += 1
            accuracy_test.append(corrects_test/len(y_test))
            
            if(epoc%50 == 0 or epoc == (EPOCH-1)):
                print("-"*8+"epoch{}".format(epoc)+"-"*8)
                print("train accuracy:{:.3}".format(accuracy_train[-1]))
                print("train loss:{:.3}".format(losses_train[-1]))
                print("test accuracy:{:.3}".format(accuracy_test[-1]))
                print("-"*20)
        return(max(accuracy_test))


    def lightGBM(self):
        lgb_train = lgb.Dataset(self.db_feature_df.values, self.db_target_df.values)
        lgb_eval = lgb.Dataset(self.query_featture_df.values, self.query_target_df.values, reference=lgb_train)

        # LightGBM parameters
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'multi_error',
            'num_class': 20,
            'learning_rate': 0.1,
            'num_leaves': 15,
            'min_data_in_leaf': 10,
            'num_iteration': 200,
            'verbose': -1,
        }
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=300,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=100)
        y_pred = gbm.predict(self.query_featture_df.values, num_iteration = gbm.best_iteration)
        y_pred_max = np.argmax(y_pred, axis=1)

        return (accuracy_score(self.query_target_df, y_pred_max))

    def pixelMatching(self):
        prediction_df = pd.DataFrame(self.query_target_df)
        for queryIndex in range(len(self.queryImages)):
            distances = np.zeros(len(self.dbImages), dtype=np.float32)
            for dbIndex in range(len(self.dbImages)):
                distances[dbIndex] = (np.abs(self.dbImages[dbIndex] - self.queryImages[queryIndex])).sum()
            minimum_id = np.argmin(distances)
            prediction_df.loc[queryIndex, "predict"] = self.db_target_df[minimum_id]
        correct_num = (prediction_df["target"] == prediction_df["predict"]).sum()
        accuracy = correct_num / prediction_df.shape[0]
        return accuracy


    def CNN(self):
        X_dbToech = torch.Tensor(self.dbImages)
        y_dbTorch = torch.LongTensor(self.db_target_df)
        X_queryTorch = torch.Tensor(self.queryImages)
        y_queryTorch = torch.LongTensor(self.query_target_df)
        
        dbDataset = TensorDataset(X_dbToech, y_dbTorch)
        queryDataset = TensorDataset(X_queryTorch, y_queryTorch)
        
        batch_size = 8
        dbLoader = DataLoader(dbDataset, batch_size=batch_size, shuffle=True)
        queryLoader = DataLoader(queryDataset, batch_size=batch_size, shuffle=False)
        conv_net = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.5),
            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            FlattenLayer()
        )
        test_input = torch.ones(1, 1, 128, 128)
        conv_output_size = conv_net(test_input).size()[-1]
        mlp = nn.Sequential(
            nn.Linear(conv_output_size, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Dropout(0.25),
            nn.Linear(200, 20)
        )
        net = nn.Sequential(
            conv_net,
            mlp
        )
        device_name = "cpu"
        net.to(device_name)
        accuracy = train_net(net, dbLoader, queryLoader, n_iter=20, device=device_name)
        return accuracy


# 評価ヘルパー
def eval_net(net, data_loader, device="cpu"):
    net.eval()
    ys = []
    ypreds = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)
    # ミニバッチ毎の結果をまとめる
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()

# 訓練ヘルパー    
def train_net(net, train_loader, test_loader,
              optimizer_cls=optim.Adam, loss_fn=nn.CrossEntropyLoss(),
              n_iter=10, device="cpu"):
    train_losses = []
    train_acc = []
    val_acc = []
    optimizer = optimizer_cls(net.parameters())
    for epoch in range(n_iter):
        running_loss = 0.0
        net.train()
        n = 0
        n_acc = 0
        for i, (xx, yy) in enumerate(tqdm(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)
            h = net(xx)
            loss = loss_fn(h, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)
            _, y_pred = h.max(1)
            n_acc += (yy == y_pred).float().sum().item()
        train_losses.append(running_loss / i)
        train_acc.append(n_acc / n)
        val_acc.append(eval_net(net, test_loader, device))
        print(epoch, train_losses[-1], train_acc[-1], val_acc[-1], flush=True)
    return(max(val_acc))
