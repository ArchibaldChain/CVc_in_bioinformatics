{
  library(readr)
  CV_errors_lmm <- read_csv("data/CV_errors_lmm.csv")
  CV_errors_ols <- read_csv("data/CV_errors_ols.csv")
  CV_errors_ridge <- read_csv("data/CV_errors_ridge.csv")
  CV_errors_wls <- read_csv("data/CV_errors_wls.csv")

  
}
{
df_lmm = CV_errors_lmm


library(reshape2)
df_9_long1 = melt(df_lmm[,c('Error_cv_lmm',  'Error_cv_lmm_c', 'Error_te_lmm')])
df_9_long2 = melt(df_lmm[,c( 'Error_cv_lmm_c', 'Error_cv_lmm')])
df_9_long3 = melt(df_lmm[,'Error_te_lmm'])

library(plyr)
mu <- ddply(df_9_long2, "variable", summarise, grp.mean=mean(value))
mu2 = mean(df_9_long3$value)
head(mu)

library(ggplot2)
 ggplot(df_9_long1, aes(x=value,color=variable)) +
    geom_density(limits=c(0.7,1.3))+
       geom_vline(xintercept = mu2, linetype = 'solid', color = 'black', size = 1.5)+
     geom_vline(data=mu, aes(xintercept=grp.mean, color=variable),
                linetype="dashed", size = 1.2)+
     ggtitle('Distribution of LMM')+
   coord_cartesian(xlim = c(0.7, 1.3))
}
 
{
  df_ols = CV_errors_ols
  
  
  library(reshape2)
  df_9_long1 = melt(df_ols[,c('Error_cv_ols',   'Error_cv_ols_c', 'Error_te_ols')])
  df_9_long2 = melt(df_ols[,c( 'Error_cv_ols_c', 'Error_cv_ols')])
  df_9_long3 = melt(df_ols[,'Error_te_ols'])
  
  library(plyr)
  mu <- ddply(df_9_long2, "variable", summarise, grp.mean=mean(value))
  mu2 = mean(df_9_long3$value)
  head(mu)
  
  library(ggplot2)
  ggplot(df_9_long1, aes(x=value,color=variable)) +
    geom_density()+
    geom_vline(xintercept = mu2, linetype = 'solid', color = 'black', size = 1.5)+
    geom_vline(data=mu, aes(xintercept=grp.mean, color=variable),
               linetype="dashed", size = 1.2)+
    ggtitle('Distribution of OLS')
}

{
  df_wls = CV_errors_wls
  
  
  library(reshape2)
  df_9_long1 = melt(df_wls[,c('Error_cv_wls',  'Error_cv_wls_c', 'Error_te_wls')])
  df_9_long2 = melt(df_wls[,c( 'Error_cv_wls_c', 'Error_cv_wls')])
  df_9_long3 = melt(df_wls[,'Error_te_wls'])
  
  library(plyr)
  mu <- ddply(df_9_long2, "variable", summarise, grp.mean=mean(value))
  mu2 = mean(df_9_long3$value)
  head(mu)
  
  library(ggplot2)
  ggplot(df_9_long1, aes(x=value,color=variable)) +
    geom_density()+
    geom_vline(xintercept = mu2, linetype = 'solid', color = 'black', size = 1.5)+
    geom_vline(data=mu, aes(xintercept=grp.mean, color=variable),
               linetype="dashed", size = 1.2)+
    ggtitle('Distribution of GLS')+
    coord_cartesian(xlim = c(0.8, 3))
}

{
  df_wls = CV_errors_wls
  
  
  library(reshape2)
  df_9_long1 = melt(df_wls[,c('Error_cv_wls',  'Error_cv_wls_c_s', 'Error_cv_wls_c', 'Error_te_wls')])
  df_9_long2 = melt(df_wls[,c( 'Error_cv_wls_c', 'Error_cv_wls', 'Error_cv_wls_c_s')])
  df_9_long3 = melt(df_wls[,'Error_te_wls'])
  
  library(plyr)
  mu <- ddply(df_9_long2, "variable", summarise, grp.mean=mean(value))
  mu2 = mean(df_9_long3$value)
  head(mu)
  
  library(ggplot2)
  ggplot(df_9_long1, aes(x=value,color=variable)) +
    geom_density()+
    geom_vline(xintercept = mu2, linetype = 'solid', color = 'black', size = 1.5)+
    geom_vline(data=mu, aes(xintercept=grp.mean, color=variable),
               linetype="dashed", size = 1.2)+
    ggtitle('Distribution For wls')
}

{
  df_ridge = CV_errors_ridge
  
  
  library(reshape2)
  df_9_long1 = melt(df_ridge[,c('Error_cv_ridge',   'Error_cv_ridge_c', 'Error_te_ridge')])
  df_9_long2 = melt(df_ridge[,c( 'Error_cv_ridge_c', 'Error_cv_ridge')])
  df_9_long3 = melt(df_ridge[,'Error_te_ridge'])
  
  library(plyr)
  mu <- ddply(df_9_long2, "variable", summarise, grp.mean=mean(value))
  mu2 = mean(df_9_long3$value)
  head(mu)
  
  library(ggplot2)
  ggplot(df_9_long1, aes(x=value,color=variable)) +
    geom_density()+
    geom_vline(xintercept = mu2, linetype = 'solid', color = 'black', size = 1.5)+
    geom_vline(data=mu, aes(xintercept=grp.mean, color=variable),
               linetype="dashed", size = 1.2)+
    ggtitle('Distribution For Ridge')
}

 