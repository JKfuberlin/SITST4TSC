BI_labels = read.csv("/home/j/data/BI/BI_labels_wrong.csv")
# transform the labels to classification labels
# Num   Name        FE                BI
# 1   Spruce        110               1
# 2   Silver fir    210               2
# 3   douglas fir   310               3
# 4   pine          410               4
# 5   oak           600-620           61, 62
# 6   red oak       630               63
# 7   beech         710               7
# 8   sycamore      820 - 829         73-75
# 9   other                           
# 1, 120 - 199, 220-299 320-399, 420-499, 500-590 (larix), 6, 690, 810, 831 - 980

# create a vector with the corresponding ranges for the categories
c0spruce = 1
c1fir = 2
c2dgl = 3
c3pine = 4
c4oak_range <- c(6,61:62)
c5redoak = 63
c6beech = 7
c7sycamore_range <- c(73:75)
c8othercon_range <- c(11:29,41,42,49,56,114,120:123)
c9otherdec_range <- c(5, 8,51,52,53,64:113,115:117,124:137)
a = BI_labels
a$encoded <- ifelse(a$ts %in% c0spruce, 0,
                    ifelse(a$ts %in% c1fir, 1,
                           ifelse(a$ts %in% c2dgl, 2,
                                  ifelse(a$ts %in% c3pine, 3,
                                         ifelse(a$ts %in% c4oak_range, 4,
                                                ifelse(a$ts %in% c5redoak, 5,
                                                       ifelse(a$ts %in% c6beech, 6,
                                                              ifelse(a$ts %in% c7sycamore_range, 7,
                                                                     ifelse(a$ts %in% c8othercon_range, 8,
                                                                            ifelse(a$ts %in% c9otherdec_range, 9, 999))))))))))

a <- dplyr::filter(a, encoded != 999) # some observations do not have a real species so i remove them
a = na.omit(a)

# check result
b = a[sample(nrow(a), 100), ]
unique(a$encoded)

hist(a$encoded)
count_1spruce = nrow(a[a$encoded == 1,])
count_2fir = nrow(a[a$encoded == 2,])
count_3dgl = nrow(a[a$encoded == 3,])
count_4pine = nrow(a[a$encoded == 4,])
count_5oak = nrow(a[a$encoded == 5,])
count_6redoak = nrow(a[a$encoded == 6,])
count_7beech = nrow(a[a$encoded == 7,])
count_8sycamore = nrow(a[a$encoded == 8,])
count_9other = nrow(a[a$encoded == 9,])

a = dplyr::select(a, -c(ts, X))

write.csv(a, '/home/j/data/BI/BI_labels_unbalanced.csv')
