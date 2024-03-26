library("plyr")
library("tidyverse")
library("sf") #   
library("data.table") # 
library("dplyr") #
# balancing training dataset for Transformer multilabel

# load each stands and read species, as well as distribution only from large ass stands files
FE = sf::st_read('/home/j/Nextcloud/Shapes/mock/mock_fe.gpkg')
stands = data.frame()
for (standnum in 1:nrow(FE)) {
  stand = FE[standnum,]
  a = dplyr::select(stand, c("OBJECTID" , "BST1_BA_1", "BST1_BA_2","BST1_BA_3", "BST1_BAA_1", "BST1_BAA_2", "BST1_BAA_3" ))
  a = sf::st_drop_geometry(a)
  stands = rbind(stands,a)
}

dictionary = read.csv('/home/j/Nextcloud/Baumartenschlüssel_FE.csv')
dict = dplyr::select(dictionary, "BST1_BA_1", "Multilabel_Encoding")
dict2 = as.list(dict)
dict1 = dict2[1] # FE codes
dict2 = dict2[2] # my encoding

# i might want to exclude stands where the three most prevalent species make up less than 90%
# stands = stands %>% filter((BST1_BAA_1+BST1_BAA_2+BST1_BAA_3)<90)
# i do not use this filter at the moment but might want to in the future so i keep it here

# i want to replace species codes by Encoded values based on the key-value pairs
for (i in 1:length(dict1[[1]])) {
  a = dict1[[1]][[i]] # selecting the nth value of the respective list
  b = dict2[[1]][[i]]
  stands$BST1_BA_1 = dplyr::case_match(stands$BST1_BA_1, a ~ b, .default = stands$BST1_BA_1)
  stands$BST1_BA_2 = dplyr::case_match(stands$BST1_BA_2, a ~ b, .default = stands$BST1_BA_2)
  stands$BST1_BA_3 = dplyr::case_match(stands$BST1_BA_3, a ~ b, .default = stands$BST1_BA_3)
}
save2 = stands

# now i need to add columns to the df representing the labels, depending on the number of target species
columnsToAdd = paste("label", 1:12,sep="")
stands[,columnsToAdd]<-NA # adding new columns
stands[is.na(stands)] <- 0 # replacing NA with 0

# if two or three species are the same, add them up
# if species amount is > 10, label for this species is 1, if > 45, replace is 2
for (i in 1:nrow(stands)) {
  # 1 & 2 the same species
  if (stands$BST1_BA_1[[i]] == stands$BST1_BA_2[[i]] & stands$BST1_BA_1[[i]] != stands$BST1_BA_3[[i]]) { # 1 == 2 but 1 != 3
    stands$BST1_BAA_1[[i]] = stands$BST1_BAA_1[[i]] + stands$BST1_BAA_2[[i]] # amount of coverage for same class is added...
    stands$BST1_BAA_2[[i]] = 0 # ...and superfluous classes deleted...
    stands$BST1_BA_2[[i]] = 0 # ...as well as their coverage
    }
  # 1 & 3 the same species
  if (stands$BST1_BA_1[[i]] == stands$BST1_BA_3[[i]] & stands$BST1_BA_1[[i]] != stands$BST1_BA_2[[i]]) {
    stands$BST1_BAA_1[[i]] = stands$BST1_BAA_1[[i]] + stands$BST1_BAA_3[[i]]
    stands$BST1_BAA_3[[i]] = 0
    stands$BST1_BA_3[[i]] = 0
  }
  
  # 2 & 3 the same species
  if (stands$BST1_BA_2[[i]] == stands$BST1_BA_3[[i]]  & stands$BST1_BA_2[[i]] != stands$BST1_BA_1[[i]]) {
    stands$BST1_BAA_2[[i]] = stands$BST1_BAA_2[[i]] + stands$BST1_BAA_3[[i]]
    stands$BST1_BAA_3[[i]] = 0
    stands$BST1_BA_3[[i]] = 0
  }
  
  # all three the same species
  if (stands$BST1_BA_1[[i]] == stands$BST1_BA_2[[i]] & stands$BST1_BA_2[[i]] == stands$BST1_BA_3[[i]]) {
    stands$BST1_BAA_1[[i]] = stands$BST1_BAA_1[[i]] + stands$BST1_BAA_2[[i]] + stands$BST1_BAA_3[[i]]
    stands$BST1_BAA_2[[i]] = 0
    stands$BST1_BAA_3[[i]] = 0
    stands$BST1_BA_2[[i]] = 0
    stands$BST1_BA_3[[i]] = 0
  }
  # hard coding!
  # encoding the position of the respective labels.
  # this changes with the amounts of classes
  pos.1 = (7 + stands$BST1_BA_1[[i]])
  pos.2 = (7 + stands$BST1_BA_2[[i]])
  pos.3 = (7 + stands$BST1_BA_3[[i]])
  
  if (stands$BST1_BAA_1[[i]]>30) {
    stands[[pos.1]][[i]] = 1
  }else {stands[[pos.1]][[i]] = 0}

  if (stands$BST1_BAA_2[[i]]>30) {
    stands[[pos.2]][[i]] = 1
  }else {stands[[pos.2]][[i]] = 0}
  
  if (stands$BST1_BAA_3[[i]]>30) {
    stands[[pos.3]][[i]] = 1
  }else {stands[[pos.3]][[i]] = 0}
}

# create labels.stands and save
multi_labels.csv = dplyr::select(stands, 1,8:ncol(stands))
write_csv(multi_labels.csv, '/home/j/data/multilabels.csv')
