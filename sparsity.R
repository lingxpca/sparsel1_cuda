library(pcaL1)
library(sparsepca) 
library(MASS) 
library(elasticnet)
library(ggplot2)
library(LaplacesDemon)
library(rospca)
library(pcaPP)
res =  data.frame(matrix(ncol = 9, nrow = 0))
colnames(res)=c( "ourdis","ourl0", "ourv","spcadis","spcal0", "spcav", "zspcadis","zspcal0", "zspcav")

for (i in (1:1000)){ 
set.seed(1)
a = si(30,10,2)
k =1 

totv = sum(diag(a$r))
spca=sparsepca::spca(a$x,k=1,alpha=i*1e-4,center=FALSE,scale=FALSE,verbose = 0)$loadings
spca = spca/sqrt(sum(spca[,k]^2))

our = sparsel1pca(a$x,projDim =1,center=FALSE,lambda=i*.1)$loadings
our = our/sqrt(sum(our[,k]^2))

zspca=elasticnet::spca(a$x,K=1,para=i*1, type="predictor",sparse="penalty")$loadings
zspca=zspca/sqrt(sum(zspca[,k]^2))

 
}
res

df1 = data.frame(x=res[,2]/10,y=res[,1],type="our")
df2 = data.frame(x=res[,5]/10,y=res[,4],type="spca")
df3 = data.frame(x=res[,8]/10,y=res[,7],type="zspca")
df = rbind(df1,df2,df3)
 
dfs <- data_summary(df, varname="y",groupnames=c("type", "x")) 
dfs$x=as.factor(dfs$x)
dfs=dfs[-which(dfs$x==0 | dfs$x ==0.9),]

ggplot(dfs,aes(x,y,group = type,colour=type)) + 
  geom_errorbar(aes(ymin=y-sd,ymax=y+sd),width=.2,position=position_dodge(width=0.2))  +  
  geom_line() + geom_point() +
  xlab("l0") + ylab("Discordance") +
  scale_color_manual(values = c("black","blue","red")) 

################
vdf1 = data.frame(x=res[,2],y=res[,3],type="our")
vdf2 = data.frame(x=res[,5],y=res[,6],type="spca")
vdf3 = data.frame(x=res[,8],y=res[,9],type="zspca")
vdf = rbind(df1,df2,df3)

vdfs <- data_summary(vdf, varname="y",groupnames=c("type", "x")) 
vdfs$x=as.factor(vdfs$x)
vdfs=vdfs[-which(vdfs$x==0 | vdfs$x ==0.9),]

ggplot(vdfs,aes(x,y,group = type,colour=type)) + 
  geom_errorbar(aes(ymin=y-sd,ymax=y+sd),width=.2,position=position_dodge(width=0.2))  +  
  geom_line() + geom_point() +
  xlab("l0") + ylab("variance") +
  scale_color_manual(values = c("black","blue","red")) 



 

###robust sparse eigenvector simulation- CROUX#### 
croux = function(n,p,delta){ 
v1 = c(sqrt(0.5),sqrt(0.5),0,0,rep(0,p-4))
v2 = c(0,0,sqrt(0.5),sqrt(0.5),rep(0,p-4))
v3 = c(sqrt(0.5),-sqrt(0.5),0,0,rep(0,p-4))
v4 = c(0,0,sqrt(0.5),-sqrt(0.5),rep(0,p-4))
v5 = rbind(matrix(0,nrow=4,ncol=p-4),diag(1,nrow=p-4,ncol=p-4))
A = cbind(v1,v2,v3,v4,v5)
E = diag(c(1,0.5,rep(0.1,p-2)))
R = A%*%E%*%t(A) #covariance
X = mvrnorm(n,rep(0,p),R) 
muout=c(2,4,2,4,rep(c(0,-1,1,0,1,-1),p))[1:p]
if (delta !=0){
outliers = mvrnorm(delta*n,muout,diag(p))
X[sample(1:n,delta*n,replace=FALSE),] =  outliers}
res = list("eigenval"=E,"eigenvec"=A,"x"=X,"cov"=R)
res
}
 

fs <- list()
for (dd in 0:4) { #different level of contamination from 0 percent to 40%
colnames(res)=c( "pca","l1pca", "rospca","ppPCA","l1pcal0", "rospcal0","pppcal0")
lres = data.frame(matrix(ncol = 7,nrow=6)) # 
idx = 1
for (l in seq(0, 100, by = 1)) { #lambda increasing
  res =  data.frame(matrix(ncol = 7, nrow = 100))
  for (i in 1:100){
  set.seed(i)
  s1 = croux(50,10,dd/10)
  mypca = prcomp(s1$x) 
  myl1pca= sparsel1pca(s1$x,projDim = 1,lambda=l,center=FALSE) 
  myrospca=rospca(s1$x,k=2,lambda=l/5,stand=FALSE) 
  mypppca = unclass(sPCAgrid(s1$x,k=1,lambda = l/10,method="sd"))
  
  pcadis = 1 - abs(t(mypca$rotation[,1])%*%s1$eigenvec[,1])
  l1pcadis=1 - abs(t(myl1pca$loadings[,1])%*%s1$eigenvec[,1])
  rospcadis = 1 - abs(t(myrospca$loadings[,1])%*%s1$eigenvec[,1])
  pppcadis = 1 -abs(t(mypppca$loadings[,1])%*%s1$eigenvec[,1])
  
  res[i,1] = pcadis
  res[i,2] = l1pcadis
  res[i,3] = rospcadis 
  res[i,4] = pppcadis
  
  res[i,5] = length(which(myl1pca$loadings[,1]==0))/length(myl1pca$loadings[,1])
  res[i,6] = length(which(myrospca$loadings[,1]==0))/length(myl1pca$loadings[,1])
  res[i,7] = length(which(mypppca$loadings[,1]==0))/length(myl1pca$loadings[,1])
  }
  lres[idx,1] = mean(res[,1])
  lres[idx,2] = mean(res[,2])
  lres[idx,3] = mean(res[,3])
  lres[idx,4] = mean(res[,4])
  lres[idx,5] = mean(res[,5]) 
  lres[idx,6] = mean(res[,6]) 
  lres[idx,7] = mean(res[,7]) 
  idx = idx + 1
}
fs[[dd+1]] = lres
}
