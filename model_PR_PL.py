# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:22:41 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 15:29:11 2021

@author: user
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import List, Dict
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
class feature_extractor(nn.Module):
    def __init__(self,hidden_1,hidden_2):
         super(feature_extractor,self).__init__()
         self.fc1=nn.Linear(310,hidden_1)
         self.fc2=nn.Linear(hidden_1,hidden_2)
         self.dropout1 = nn.Dropout(p=0.25)
         self.dropout2 = nn.Dropout(p=0.25)
    def forward(self,x):
         x=self.fc1(x)
         x=F.relu(x)
#         x=F.leaky_relu(x)
         x=self.fc2(x)
         x=F.relu(x)
#         x=F.leaky_relu(x)
         return x
    def get_parameters(self) -> List[Dict]:
         params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
                  ]
         return params  
class discriminator(nn.Module):
    def __init__(self,hidden_1):
         super(discriminator,self).__init__()
         self.fc1=nn.Linear(hidden_1,hidden_1)
         self.fc2=nn.Linear(hidden_1,1)
         self.dropout1 = nn.Dropout(p=0.25)
         self.sigmoid = nn.Sigmoid()
    def forward(self,x):
         x=self.fc1(x)
         x=F.relu(x)
#         x=F.leaky_relu(x)
         x=self.dropout1(x)
         x=self.fc2(x)
         x=self.sigmoid(x)
         return x
    def get_parameters(self) -> List[Dict]:
         params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
                  ]
         return params 
class Domain_adaption_model(nn.Module):
   def __init__(self,hidden_1,hidden_2,hidden_3,hidden_4,num_of_class,low_rank,max_iter,upper_threshold,lower_threshold):
       super(Domain_adaption_model,self).__init__()
       self.fea_extrator_f= feature_extractor(hidden_1,hidden_2)
       self.fea_extrator_g= feature_extractor(hidden_3,hidden_4)
       self.U=nn.Parameter(torch.randn(low_rank,hidden_2),requires_grad=True)
       self.V=nn.Parameter(torch.randn(low_rank,hidden_4),requires_grad=True)
       self.P=torch.randn(num_of_class,hidden_4)
       self.stored_mat=torch.matmul(self.V,self.P.T)
       self.max_iter=max_iter
       self.upper_threshold=upper_threshold
       self.lower_threshold=lower_threshold
#       self.diff=(upper_threshold-lower_threshold)
       self.threshold=upper_threshold
       self.cluster_label=np.zeros(num_of_class)
       self.num_of_class=num_of_class
   def forward(self,source,target,source_label):
       feature_source_f=self.fea_extrator_f(source)
       feature_target_f=self.fea_extrator_f(target)
#       feature_source_g=feature_source_f
       feature_source_g=self.fea_extrator_f(source)
       ##torch.matmul(source_label.T,torch.ones(batch_num,num_of_class))
       ## Update P through some algebra computations for the convenice of broadcast
       self.P=torch.matmul(torch.inverse(torch.diag(source_label.sum(axis=0))+torch.eye(self.num_of_class).cuda()),torch.matmul(source_label.T,feature_source_g))
#       self.P=torch.matmul(torch.inverse(torch.diag(source_label.sum(axis=0))),torch.matmul(source_label.T,feature_source_g))
       self.stored_mat=torch.matmul(self.V,self.P.T)
       source_predict=torch.matmul(torch.matmul(self.U,feature_source_f.T).T,self.stored_mat)
       target_predict=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat)
#       source_logit  =source_predict
       source_label_feature=torch.nn.functional.softmax(source_predict, dim=1)
       target_label_feature=torch.nn.functional.softmax(target_predict, dim=1)
       ## DAC part
       sim_matrix=self.get_cos_similarity_distance(source_label_feature)
       sim_matrix_target=self.get_cos_similarity_distance(target_label_feature)
       return source_predict,feature_source_f,feature_target_f,sim_matrix,sim_matrix_target
   def compute_target_centroid(self,target,target_label):
       feature_source_g=self.fea_extrator_f(target)
       target_centroid=torch.matmul(torch.inverse(torch.diag(target_label.sum(axis=0))+torch.eye(self.num_of_class).cuda()),torch.matmul(target_label.T,feature_source_g))
       return target_centroid
   def target_domain_evaluation(self,test_features,test_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(test_features)
       test_logit=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat.cuda())
       test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
       test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)
       test_labels=np.argmax(test_labels.cpu().detach().numpy(),axis=1)
       test_predict=np.zeros_like(test_labels)
       for i in range(len(self.cluster_label)):
           cluster_index=np.where(test_cluster==i)[0]
           test_predict[cluster_index]=self.cluster_label[i]
#       acc=np.sum(label_smooth(test_predict)==test_labels)/len(test_predict)
       acc=np.sum(test_predict==test_labels)/len(test_predict)
       nmi=metrics.normalized_mutual_info_score(test_predict,test_labels)
       return acc,nmi   
   def cluster_label_update(self,source_features,source_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(source_features)
       source_logit=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat.cuda())
       source_cluster=np.argmax(torch.nn.functional.softmax(source_logit, dim=1).cpu().detach().numpy(),axis=1)
       source_labels=np.argmax(source_labels.cpu().detach().numpy(),axis=1)
       for i in range(len(self.cluster_label)):
           samples_in_cluster_index=np.where(source_cluster==i)[0]
           label_for_samples=source_labels[samples_in_cluster_index]
           if len(label_for_samples)==0:
              self.cluster_label[i]=0
           else:
              label_for_current_cluster=np.argmax(np.bincount(label_for_samples))
              self.cluster_label[i]=label_for_current_cluster
       source_predict=np.zeros_like(source_labels)
       for i in range(len(self.cluster_label)):
           cluster_index=np.where(source_cluster==i)[0]
           source_predict[cluster_index]=self.cluster_label[i]
       acc=np.sum(source_predict==source_labels)/len(source_predict)
       nmi=metrics.normalized_mutual_info_score(source_predict,source_labels)
       return acc,nmi
   def visualization(self,target,target_labels,tsne=0):
       feature_target_f=self.fea_extrator_f(target)
       target_feature=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat.cuda())
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       feature_target_f=feature_target_f.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]]
       # 画散点图
           fig = plt.figure()
           ax = Axes3D(fig)
           ax.scatter(x0[:,0],x0[:,1],x0[:,2], s=area, c=colors1, alpha=0.4)
           ax.scatter(x1[:,0],x1[:,1],x1[:,2], s=area, c=colors2, alpha=0.4)
           ax.scatter(x2[:,0],x2[:,1],x2[:,2], s=area, c=colors3, alpha=0.4)
           plt.show()
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(feature_target_f.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.show()
   def visualization_4(self,target,target_labels,tsne=0):
       target_feature=self.fea_extrator_f(target)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       colors4 = '#000000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           print('error')
           return
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(target_feature.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           x3=target_feature[np.where(target_labels==3)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.scatter(x3[:,0],x3[:,1], s=area, c=colors4, alpha=0.4)
           plt.show()
   def predict(self,target):
       with torch.no_grad():
           self.eval()         
           feature_target_f=self.fea_extrator_f(target)
           test_logit=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat.cuda())/8
           test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
           test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)
           cluster_0_index,cluster_1_index,cluster_2_index=np.where(test_cluster==0)[0],np.where(test_cluster==1)[0],np.where(test_cluster==2)[0]
           test_cluster[cluster_0_index]=self.cluster_label[0]
           test_cluster[cluster_1_index]=self.cluster_label[1]
           test_cluster[cluster_2_index]=self.cluster_label[2]
       return test_cluster
   def get_cos_similarity_distance(self, features):
        """Get distance in cosine similarity
        :param features: features of samples, (batch_size, num_clusters)
        :return: distance matrix between features, (batch_size, batch_size)
        """
        # (batch_size, num_clusters)
        features_norm = torch.norm(features, dim=1, keepdim=True)
        # (batch_size, num_clusters)
        features = features / features_norm
        # (batch_size, batch_size)
        cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
        return cos_dist_matrix
   def get_cos_similarity_by_threshold(self, cos_dist_matrix):
        """Get similarity by threshold
        :param cos_dist_matrix: cosine distance in matrix,
        (batch_size, batch_size)
        :param threshold: threshold, scalar
        :return: distance matrix between features, (batch_size, batch_size)
        """
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        similar = torch.tensor(1, dtype=dtype, device=device)
        dissimilar = torch.tensor(0, dtype=dtype, device=device)
        sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar,
                                 dissimilar)
        return sim_matrix
   def compute_indicator(self,cos_dist_matrix):
       device = cos_dist_matrix.device
       dtype = cos_dist_matrix.dtype
       selected = torch.tensor(1, dtype=dtype, device=device)
       not_selected = torch.tensor(0, dtype=dtype, device=device)
       w2=torch.where(cos_dist_matrix < self.lower_threshold,selected,not_selected)
       w1=torch.where(cos_dist_matrix > self.upper_threshold,selected,not_selected)
       w = w1 + w2
       nb_selected=torch.sum(w)
       return w,nb_selected
   def update_threshold(self, epoch: int):
        """Update threshold
        :param threshold: scalar
        :param epoch: scalar
        :return: new_threshold: scalar
        """
        n_epochs = self.max_iter
        diff = self.upper_threshold - self.lower_threshold
        eta = diff / n_epochs
#        eta=self.diff/ n_epochs /2
        # First epoch doesn't update threshold
        if epoch != 0:
            self.upper_threshold = self.upper_threshold-eta
            self.lower_threshold = self.lower_threshold+eta
        else:
            self.upper_threshold = self.upper_threshold
            self.lower_threshold = self.lower_threshold
        self.threshold=(self.upper_threshold+self.lower_threshold)/2
#        print(">>> new threshold is {}".format(new_threshold), flush=True)
   def get_parameters(self) -> List[Dict]:
       params = [
            {"params": self.fea_extrator_f.fc1.parameters(), "lr_mult": 1},
            {"params": self.fea_extrator_f.fc2.parameters(), "lr_mult": 1},
            {"params": self.fea_extrator_g.fc1.parameters(), "lr_mult": 1},
            {"params": self.fea_extrator_g.fc2.parameters(), "lr_mult": 1},
            {"params": self.U, "lr_mult": 1},
            {"params": self.V, "lr_mult": 1},
                ]
       return params  
   
class Domain_adaption_model_withoutproto(nn.Module):
   def __init__(self,hidden_1,hidden_2,hidden_3,hidden_4,num_of_class,low_rank,max_iter,upper_threshold,lower_threshold):
       super(Domain_adaption_model_withoutproto,self).__init__()
       self.fea_extrator_f= feature_extractor(hidden_1,hidden_2)
       self.max_iter=max_iter
       self.upper_threshold=upper_threshold
       self.lower_threshold=lower_threshold
       self.classifier=nn.Linear(hidden_2,num_of_class)
#       self.diff=(upper_threshold-lower_threshold)
       self.threshold=upper_threshold
       self.cluster_label=np.zeros(num_of_class)
       self.num_of_class=num_of_class
   def forward(self,source,target,source_label):
       feature_source_f=self.fea_extrator_f(source)
       feature_target_f=self.fea_extrator_f(target)
       ##torch.matmul(source_label.T,torch.ones(batch_num,num_of_class))
       ## Update P through some algebra computations for the convenice of broadcast
       source_predict=self.classifier(feature_source_f)
       target_predict=self.classifier(feature_target_f)
#       source_logit  =source_predict
       source_label_feature=torch.nn.functional.softmax(source_predict, dim=1)
       target_label_feature=torch.nn.functional.softmax(target_predict, dim=1)
       ## DAC part
       sim_matrix=self.get_cos_similarity_distance(source_label_feature)
       sim_matrix_target=self.get_cos_similarity_distance(target_label_feature)
       return source_predict,feature_source_f,feature_target_f,sim_matrix,sim_matrix_target
   def compute_target_centroid(self,target,target_label):
       feature_source_g=self.fea_extrator_f(target)
       target_centroid=torch.matmul(torch.inverse(torch.diag(target_label.sum(axis=0))+torch.eye(self.num_of_class).cuda()),torch.matmul(target_label.T,feature_source_g))
       return target_centroid
   def target_domain_evaluation(self,test_features,test_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(test_features)
       test_logit=self.classifier(feature_target_f)
       test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
       test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)
       test_labels=np.argmax(test_labels.cpu().detach().numpy(),axis=1)
       test_predict=np.zeros_like(test_labels)
       for i in range(len(self.cluster_label)):
           cluster_index=np.where(test_cluster==i)[0]
           test_predict[cluster_index]=self.cluster_label[i]
#       acc=np.sum(label_smooth(test_predict)==test_labels)/len(test_predict)
       acc=np.sum(test_predict==test_labels)/len(test_predict)
       nmi=metrics.normalized_mutual_info_score(test_predict,test_labels)
       return acc,nmi   
   def cluster_label_update(self,source_features,source_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(source_features)
       source_logit=self.classifier(feature_target_f)
       source_cluster=np.argmax(torch.nn.functional.softmax(source_logit, dim=1).cpu().detach().numpy(),axis=1)
       source_labels=np.argmax(source_labels.cpu().detach().numpy(),axis=1)
       for i in range(len(self.cluster_label)):
           samples_in_cluster_index=np.where(source_cluster==i)[0]
           label_for_samples=source_labels[samples_in_cluster_index]
           if len(label_for_samples)==0:
              self.cluster_label[i]=0
           else:
              label_for_current_cluster=np.argmax(np.bincount(label_for_samples))
              self.cluster_label[i]=label_for_current_cluster
       source_predict=np.zeros_like(source_labels)
       for i in range(len(self.cluster_label)):
           cluster_index=np.where(source_cluster==i)[0]
           source_predict[cluster_index]=self.cluster_label[i]
       acc=np.sum(source_predict==source_labels)/len(source_predict)
       nmi=metrics.normalized_mutual_info_score(source_predict,source_labels)
       return acc,nmi
   def visualization(self,target,target_labels,tsne=0):
       feature_target_f=self.fea_extrator_f(target)
       target_feature=self.classifier(feature_target_f)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       feature_target_f=feature_target_f.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]]
       # 画散点图
           fig = plt.figure()
           ax = Axes3D(fig)
           ax.scatter(x0[:,0],x0[:,1],x0[:,2], s=area, c=colors1, alpha=0.4)
           ax.scatter(x1[:,0],x1[:,1],x1[:,2], s=area, c=colors2, alpha=0.4)
           ax.scatter(x2[:,0],x2[:,1],x2[:,2], s=area, c=colors3, alpha=0.4)
           plt.show()
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(feature_target_f.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.show()
   def visualization_4(self,target,target_labels,tsne=0):
       target_feature=self.fea_extrator_f(target)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       colors4 = '#000000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           print('error')
           return
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(target_feature.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           x3=target_feature[np.where(target_labels==3)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.scatter(x3[:,0],x3[:,1], s=area, c=colors4, alpha=0.4)
           plt.show()
   def get_cos_similarity_distance(self, features):
        """Get distance in cosine similarity
        :param features: features of samples, (batch_size, num_clusters)
        :return: distance matrix between features, (batch_size, batch_size)
        """
        # (batch_size, num_clusters)
        features_norm = torch.norm(features, dim=1, keepdim=True)
        # (batch_size, num_clusters)
        features = features / features_norm
        # (batch_size, batch_size)
        cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
        return cos_dist_matrix
   def get_cos_similarity_by_threshold(self, cos_dist_matrix):
        """Get similarity by threshold
        :param cos_dist_matrix: cosine distance in matrix,
        (batch_size, batch_size)
        :param threshold: threshold, scalar
        :return: distance matrix between features, (batch_size, batch_size)
        """
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        similar = torch.tensor(1, dtype=dtype, device=device)
        dissimilar = torch.tensor(0, dtype=dtype, device=device)
        sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar,
                                 dissimilar)
        return sim_matrix
   def compute_indicator(self,cos_dist_matrix):
       device = cos_dist_matrix.device
       dtype = cos_dist_matrix.dtype
       selected = torch.tensor(1, dtype=dtype, device=device)
       not_selected = torch.tensor(0, dtype=dtype, device=device)
       w2=torch.where(cos_dist_matrix < self.lower_threshold,selected,not_selected)
       w1=torch.where(cos_dist_matrix > self.upper_threshold,selected,not_selected)
       w = w1 + w2
       nb_selected=torch.sum(w)
       return w,nb_selected
   def update_threshold(self, epoch: int):
        """Update threshold
        :param threshold: scalar
        :param epoch: scalar
        :return: new_threshold: scalar
        """
        n_epochs = self.max_iter
        diff = self.upper_threshold - self.lower_threshold
        eta = diff / n_epochs
#        eta=self.diff/ n_epochs /2
        # First epoch doesn't update threshold
        if epoch != 0:
            self.upper_threshold = self.upper_threshold-eta
            self.lower_threshold = self.lower_threshold+eta
        else:
            self.upper_threshold = self.upper_threshold
            self.lower_threshold = self.lower_threshold
        self.threshold=(self.upper_threshold+self.lower_threshold)/2
#        print(">>> new threshold is {}".format(new_threshold), flush=True)
   def get_parameters(self) -> List[Dict]:
       params = [
            {"params": self.fea_extrator_f.fc1.parameters(), "lr_mult": 1},
            {"params": self.fea_extrator_f.fc2.parameters(), "lr_mult": 1},
            {"params": self.classifier.parameters(), "lr_mult": 1},
                ]
       return params
