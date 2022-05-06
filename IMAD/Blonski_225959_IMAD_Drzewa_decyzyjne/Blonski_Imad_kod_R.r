options(warn=-1) #Wyłączenie warningów aby w sprawozdaniu niebyły drukowane
library(tidyverse)
library(C50) #Drzewa decyzyjne C5.0
library(caret)#Pakiet do uczenia maszynowego
library(MLmetrics)# Pakiet zawierający metryki takie jak Fscore, Precision itp.
library(rattle) #fency plot
options(repr.plot.width=8, repr.plot.height=3)



iris_data = read.csv(file = "iris.csv") #załaduj do iris_data dane Iris.
head(iris_data,3) # 3 pierwsze rekordy z datasetu

wine_data = read.csv(file = "wine.csv") #załaduj do wine_data dane Wine Quality
head(wine_data,3)

glass_data = read.csv(file = "glass.csv") #załąduj do glass_data dane Glass
head(glass_data,3)

seed_data = read.csv(file = "seeds.csv") #załaduj do seed_data dane Seeds
head(seed_data,3)

column_to_drop <- c("compactness")
seed_data <- seed_data[ , !(names(seed_data) %in% column_to_drop)]

metrics <- function(data, lev = NULL, model = NULL) 
{
  f1_val <- F1_Score(y_pred = data$pred, y_true = data$obs, positive = lev[1])  
  rec_val <-Recall(y_pred = data$pred, y_true = data$obs, positive = lev[1])
  sen_val = Sensitivity(y_pred = data$pred, y_true = data$obs, positive = lev[1])
  pre_val = Precision(y_pred = data$pred, y_true = data$obs, positive = lev[1])
  acc_val = Accuracy(y_pred = data$pred, y_true = data$obs)  
  c(fScore = f1_val,Recall = rec_val, Sensitivity = sen_val, Precision = pre_val,Accuracy=acc_val)
}

TreeModel_caret <- function(param_names,dataset,model_type,starting_col_number,last_col_number,formula,folds,winnowing,fuzzy,GlobalPruning
){
    #Selekcja danych
    test = dataset$last_col_number
    dataset[,ncol(dataset)] = as.factor(dataset[,ncol(dataset)])
    y = dataset[,(last_col_number-1)] #class column
    index = createDataPartition(y=y, p=0.7, list=FALSE)
    train.set = dataset[index,starting_col_number:last_col_number]
    test.set = dataset[-index,starting_col_number:last_col_number]
    #Ustawianie parametrów i Control
    train.control <- trainControl(#https://www.rdocumentation.org/packages/C50/versions/0.1.3/topics/C5.0Control
                  method = "cv",
                  number = folds,
                  savePredictions = "all",
                  summaryFunction = metrics)
    
    Control <- C5.0Control(
                  winnow = winnowing,
                  fuzzyThreshold = fuzzy,
                  noGlobalPruning = GlobalPruning)
    #uczenie
    tree <- train(
                  formula,
                  data=train.set, 
                  method="C5.0",
                  control = Control,
                  tuneGrid = data.frame(trials = 1, model = c(model_type), winnow = winnowing),
                  trControl = train.control)
    #wyciąganie metryk z results
    f1 = tree$results$fScore
    rec = tree$results$Recall
    sen = tree$results$Sensitivity
    acc = tree$results$Accuracy
    prec= tree$results$Precision
    size = tree$finalModel$size
    # przygotowanie danych do zwrócenia
    research_frame<-data.frame(param_names,f1,acc,rec,prec,size)
    names(research_frame)<-c("params","f1","acc","rec","prec","Tree_size")
    return(research_frame)
}

options(repr.plot.width=15, repr.plot.height=10)
in_train <- as.factor(sample(1:nrow(iris_data), size = (0.8*nrow(iris_data))))
train_data <- iris_data[ in_train,]
test_data  <- iris_data[-in_train,]
tree_mod_iris <- C5.0(x = train_data[, 2:5], y = train_data[,6])
plot(tree_mod_iris)
print(paste("Tree size is: ", tree_mod_iris$size))

in_train <- as.factor(sample(1:nrow(glass_data), size = (0.8*nrow(glass_data))))
glass_data[,ncol(glass_data)] <- as.factor(glass_data[,ncol(glass_data)])
train_data <- glass_data[ in_train,]
test_data  <- glass_data[-in_train,]
tree_mod_glass <- C5.0(x = train_data[, 1:9], y = train_data[,10])
plot(tree_mod_glass)
print(paste("Tree size is: ", tree_mod_iris$size))

in_train <- as.factor(sample(1:nrow(wine_data), size = (0.8*nrow(wine_data))))
wine_data[,ncol(wine_data)] <- as.factor(wine_data[,ncol(wine_data)])
train_data <- wine_data[ in_train,]
test_data  <- wine_data[-in_train,]
tree_mod_wine <- C5.0(x = train_data[, 1:11], y = train_data[,12])
plot(tree_mod_wine)
print(tree_mod_wine$size)

in_train <- as.factor(sample(1:nrow(seed_data), size = (0.8*nrow(seed_data))))
seed_data[,ncol(seed_data)] <- as.factor(seed_data[,ncol(seed_data)])
train_data <- seed_data[ in_train,]
test_data  <- seed_data[-in_train,]
tree_mod_seed <- C5.0(x = train_data[, 1:6], y = train_data[,7])
plot(tree_mod_seed)
print(paste("Tree size is: ", tree_mod_iris$size))

#Sprawdzone parametry dla danego datasetu
research <- function(dataset,start_col,end_col,formula){
    research_dataframe<-data.frame()
    #Params = FALSE
    newMeasureDataframe = TreeModel_caret("Tree F=5 winnow=F fuzzy=F  NoPruning=F",dataset,"tree",start_col,end_col,formula,5,FALSE,FALSE,FALSE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    #Winnowing TRUE
    newMeasureDataframe = TreeModel_caret("Tree F=5 winnow=T fuzzy=F  NoPruning=F",dataset,"tree",start_col,end_col,formula,5,TRUE,FALSE,FALSE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    #fuzzyThreshold TRUE
    newMeasureDataframe = TreeModel_caret("Tree F=5 winnow=F fuzzy=T  NoPruning=F",dataset,"tree",start_col,end_col,formula,5,FALSE,TRUE,FALSE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    # pruning TRUE
    newMeasureDataframe = TreeModel_caret("Tree F=5 winnow=F fuzzy=F  NoPruning=T",dataset,"tree",start_col,end_col,formula,5,FALSE,FALSE,TRUE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    #Winnowing True & fuzzyThreshold True
    newMeasureDataframe = TreeModel_caret("Tree F=5 winnow=T fuzzy=T  NoPruning=F",dataset,"tree",start_col,end_col,formula,5,TRUE,TRUE,FALSE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    #Winnowing True & noGlobalPrunning True
    newMeasureDataframe = TreeModel_caret("Tree F=5 winnow=T fuzzy=F  NoPruning=T",dataset,"tree",start_col,end_col,formula,5,TRUE,FALSE,TRUE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    #noGlobalPrunning True fuzzyThreshold True
    newMeasureDataframe = TreeModel_caret("Tree F=5 winnow=F fuzzy=T  NoPruning=T",dataset,"tree",start_col,end_col,formula,5,FALSE,TRUE,TRUE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    #All params true
    newMeasureDataframe = TreeModel_caret("Tree F=5 winnow=T fuzzy=T  NoPruning=T",dataset,"tree",start_col,end_col,formula,5,TRUE,TRUE,TRUE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    
    #FOLDS 10
    #Params False
    newMeasureDataframe = TreeModel_caret("Tree F=10 winnow=F fuzzy=F  NoPruning=F",dataset,"tree",start_col,end_col,formula,10,FALSE,FALSE,FALSE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
   
    #Winnowing
    newMeasureDataframe = TreeModel_caret("Tree F=10 winnow=T fuzzy=F  NoPruning=F",dataset,"tree",start_col,end_col,formula,10,TRUE,FALSE,FALSE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    
    #fuzzyThredshold
    newMeasureDataframe = TreeModel_caret("Tree F=10 winnow=F fuzzy=T  NoPruning=F",dataset,"tree",start_col,end_col,formula,10,FALSE,TRUE,FALSE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    
    #Pruning
    newMeasureDataframe = TreeModel_caret("Tree F=10 winnow=F fuzzy=F  NoPruning=T",dataset,"tree",start_col,end_col,formula,10,FALSE,FALSE,TRUE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    
    #Winnowing True & fuzzyThreshold True
    newMeasureDataframe = TreeModel_caret("Tree F=10 winnow=T fuzzy=T  NoPruning=F",dataset,"tree",start_col,end_col,formula,10,TRUE,TRUE,FALSE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    
    #Winnowing True & noGlobalPrunning True
    newMeasureDataframe = TreeModel_caret("Tree F=10 winnow=T fuzzy=F  NoPruning=T",dataset,"tree",start_col,end_col,formula,10,TRUE,FALSE,TRUE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    
    #noGlobalPrunning True fuzzyThreshold True
    newMeasureDataframe = TreeModel_caret("Tree F=10 winnow=F fuzzy=T  NoPruning=T",dataset,"tree",start_col,end_col,formula,10,FALSE,TRUE,TRUE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
   
    #All params true
    newMeasureDataframe = TreeModel_caret("Tree F=10 winnow=T fuzzy=T  NoPruning=T",dataset,"tree",start_col,end_col,formula,10,TRUE,TRUE,TRUE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
   
    #Params False
    newMeasureDataframe = TreeModel_caret("Tree F=15 winnow=F fuzzy=F  NoPruning=F",dataset,"tree",start_col,end_col,formula,15,FALSE,FALSE,FALSE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    #Winnowing
    newMeasureDataframe = TreeModel_caret("Tree F=15 winnow=T fuzzy=F  NoPruning=F",dataset,"tree",start_col,end_col,formula,15,TRUE,FALSE,FALSE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    #fuzzyThredshold
    newMeasureDataframe = TreeModel_caret("Tree F=15 winnow=F fuzzy=T  NoPruning=F",dataset,"tree",start_col,end_col,formula,15,FALSE,TRUE,FALSE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    #Pruning
    newMeasureDataframe = TreeModel_caret("Tree F=15 winnow=F fuzzy=F  NoPruning=T",dataset,"tree",start_col,end_col,formula,15,FALSE,FALSE,TRUE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    #Winnowing True & fuzzyThreshold True
    newMeasureDataframe = TreeModel_caret("Tree F=15 winnow=T fuzzy=T  NoPruning=F",dataset,"tree",start_col,end_col,formula,15,TRUE,TRUE,FALSE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    #Winnowing True & noGlobalPrunning True
    newMeasureDataframe = TreeModel_caret("Tree F=15 winnow=T fuzzy=F  NoPruning=T",dataset,"tree",start_col,end_col,formula,15,TRUE,FALSE,TRUE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
     #noGlobalPrunning True fuzzyThreshold True
    newMeasureDataframe = TreeModel_caret("Tree F=15 winnow=F fuzzy=T  NoPruning=T",dataset,"tree",start_col,end_col,formula,15,FALSE,TRUE,TRUE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
     #All params true
    newMeasureDataframe = TreeModel_caret("Tree F=15 winnow=T fuzzy=T  NoPruning=T",dataset,"tree",start_col,end_col,formula,15,TRUE,TRUE,TRUE)
    research_dataframe <- rbind(research_dataframe, newMeasureDataframe)
    
    #f1 plot
    library(RColorBrewer)
    coul <- brewer.pal(5, "Set2") 
    par(mar=c(4,19,4,4))
    barplot(height=research_dataframe$f1,
            names=research_dataframe$params,
            col=coul,
            horiz=T,
            width=2,
            las=2,
            xlab="Fscore",)
    return(research_dataframe)
}



print('Iris')
research(iris_data,2,6,Species~.)

print('Seeds')
research(seed_data,1,7,Type~.)

print('Glass')
research(glass_data,1,10,Type~. )

print('Wine quality')
research(wine_data,1,12,quality~.)



