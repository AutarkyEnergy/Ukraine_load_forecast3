library(MuMIn)

## set working directory and check if it is correct
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()


## load data----
MT_deterministic_data <- read.csv("./Data/mid_deterministic_data.csv")

# make a training set until end of 2018
mid_training = MT_deterministic_data[1:2190,]


######## Find best linear models  ######## 
## PLEASE NOTE THAT CALCULATION TAKES MORE THAN 12 HOURS, RESULTS ARE ALREADY CALCULATED AND
## CAN BE LOADED IN THE NEXT SECTION.


## create global model with all midterm indicators except type of month
variables_midterm=colnames(MT_deterministic_data[,c(5:17,29:36)])

mt_formula <- as.formula(paste("mid_load", paste(variables_midterm, collapse = " + "), 
                               sep = " ~ "))

global_mid_model <- lm(mt_formula, data= mid_training,na.action="na.fail")

## create models with all combinations of all indicators
# will take a very long time - depending on the available computing power around 12-24 hours
combinations_mid <- dredge(global_mid_model)

write.csv(combinations_mid,"./Data/midterm_combinations_final.csv",row.names = F)


### LOAD RESULTS IN THIS SECTION TO SKIP 12h COMPUTATION TIME  
#   Continue with evaluating the model results ----

combination_mid_final <-read.csv("./Data/midterm_combinations_final.csv")

## manually inspect the best models (top of the list) 
# --> Models 1-4 have the same AICc, model 4 has the best predictors from an empirical point of view.  

variables_mid <- colnames(combination_mid_final[4,2:22])[complete.cases(t(combination_mid_final[4,2:22]))]

## add month variables step-wise
variables_midterm_month=colnames(MT_deterministic_data[,c(18:28)])


month_formula <- as.formula(paste("mid_load", paste(variables_midterm_month, collapse = " + "), 
                               sep = " ~ "))

model_month <- lm(month_formula, data= mid_training,na.action="na.fail")

# dredge() method is just a quick way to get the logical matrix, should go very fast
combinations_month <- dredge(model_month,trace = T)

# initiate a data frame to save the model AICc
rank_df <- as.data.frame(matrix(nrow=nrow(combinations_month),ncol = 2))
colnames(rank_df)<- c("AICc","model_no")

# calculate all combinations with month added and save AICc

for (i in 1:nrow(combinations_month)){
variables_month=colnames(combinations_month[i,2:12])[complete.cases(t(combinations_month[i,2:12]))]
variables_all= c(variables_mid,variables_month)
formula_all <- as.formula(paste("mid_load", paste(variables_all, collapse = " + "), 
                                  sep = " ~ "))
model_all <- lm(formula_all, data= mid_training,na.action="na.fail")

rank_df$AICc[i] <- AICc(model_all)
rank_df$model_no[i] <- i
}

# look at best models
best_models <- order(rank_df$AICc)

variables_best_month=colnames(combinations_month[best_models[1],2:12])[complete.cases(t(combinations_month[best_models[1],2:12]))]
variables_final= c(variables_mid,variables_best_month)
formula_final <- as.formula(paste("mid_load", paste(variables_final, collapse = " + "), 
                                sep = " ~ "))
model_final <- lm(formula_final, data= mid_training,na.action="na.fail")
save(model_final,file = "./Models/midterm_deterministic/model_final.Rdata")

### Linear regressive mid-term model is done, residuals are calculated now for the LSTM and ARIMA models 


MT <- predict(model_final,MT_deterministic_data, interval = "confidence")

residuals <- MT_deterministic_data$mid_load - MT[,1]
MT_deterministic_data$target_residuals <- residuals
MT_deterministic_data$lm_pred <- MT[,1]

### save dataframe for LSTM script
write.csv(MT_deterministic_data,"./Data/midterm_ML.csv", row.names = F)




###### ARIMA calculation ######## 

# load deterministic model and residuals
load("./Models/midterm_deterministic/model_final.Rdata")

library(tseries)
library(forecast)

res <- model_final$residuals
acf(res)
pacf(res)  
adf.test(res)
kpss.test(res)
# --> data is non-stationary
# double-check with auto.arima()
starting_model <- auto.arima(diff_res)
checkresiduals(starting_model)
# --> residuals need to be differenced once to become stationary
d=1
diff_res <- diff(res,1)
acf(diff_res)  
# last significant lag at t=7  --> max grid search p-oder = 9
pacf(diff_res) 
# last significant lat at t=25 --> max grid search q-oder = 27

test_df<- data.frame(matrix(nrow=(9*27),ncol = 3))
colnames(test_df)<- c("AICC","sum_nine","sum_all")

for (k in 1:27){
  q=k
  for (i in 1:9){
    tryCatch({
    print(paste('processing ARIMA:',i,d,q))
    grid_model <- Arima(res, order = c(i, d, q))
    forecasted_arima<-forecast(grid_model, h=730,biasadj=TRUE,bootstrap = TRUE)
    test_df$AICC[((k-1)*9+i)] <- AICc(grid_model)      
    res_forecasted <- forecasted_arima$mean
    res_forecasted_nine <- forecasted_arima$mean[1:9] 
    res_forecasted_sum <- MT_deterministic_data$target_residuals[2191:2920] - res_forecasted
    
    test_df$sum_all[((k-1)*9+i)]=sum(abs(res_forecasted_sum))/sum(abs(MT_deterministic_data$target_residuals[2191:2920]))
    
    res_forecasted_sum_nine <- MT_deterministic_data$target_residuals[2191:2199] - res_forecasted_nine
    test_df$sum_nine[((k-1)*9+i)]=sum(abs(res_forecasted_sum_nine))/sum(abs(MT_deterministic_data$target_residuals[2191:2199]))
    
    
    
    },error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    
  }}





for (i in 1:10) {
  tryCatch({
    print(i)
    if (i==7) stop("Urgh, the iphone is in the blender !")
  }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
}






 
starting_model <- auto.arima(diff_res)
checkresiduals(starting_model)
AICc(starting_model)

q <- length(starting_model$model$theta)
d <- starting_model$model$Delta
if(is.null(d)==TRUE){
  d<-0
}


### grid search for best ARIMA based on AICc and prediction accuracy for future values 
#   taking the d value given by auto.arima(), focusing on p and q

model_aic= data.frame()[1:2000,]
model_aic$AICc <- 100000
model_pred_acc= data.frame()[1:2000, ]
model_pred_acc$sum_errors <- 10000000

plotdf$fitted_stoch_arima_noreg <-0 
#plotdf$fitted_stoch_arima_noreg [1:2190]<-fitted(arima_noreg)
for (k in 24:28){ q=k
for (i in 1:15){
  print(paste('processing ARIMA:',i,d,q))
  grid_model <- Arima(res, order = c(i, d, q))
  forecasted_arima_noreg<-forecast(grid_model, h=730,biasadj=TRUE,bootstrap = TRUE)
  
  plotdf$fitted_stoch_arima_noreg [2191:2920]<- forecasted_arima_noreg$mean
  model_pred_acc[((k-1)*15+i),] <-sum(abs(plotdf$res[2191:2920]-plotdf$fitted_stoch_arima_noreg [2191:2920]))
  
  print(model_pred_acc[((k-1)*15+i),])
  model_aic[((k-1)*15+i),] <- AICc(grid_model)
}}








