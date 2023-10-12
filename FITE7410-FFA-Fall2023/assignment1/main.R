# Load necessary libraries
#remove ggplot2
remove.packages("ggplot2")

#install ggplot2
install.packages("ggplot2")
install.packages("corrplot")
install.packages("gridExtra")

#load ggplot2
library(ggplot2)
library(gridExtra)
library(corrplot)

# Load the dataset
enron_data <- read.csv("/Users/user/Downloads/git_clone/HKU-MSc-CS-Resource/FITE7410-FFA-Fall2023/assignment1/enron.csv")
summary(enron_data)


library(ggplot2)

# Univariate analysis for numerical columns
numerical_cols <- names(enron_data)[sapply(enron_data, is.numeric)]

# Plot histograms for numerical columns
par(mfrow=c(4, 5))
for (col in numerical_cols) {
  hist(enron_data[[col]], main=col, xlab=col, breaks=30, col="lightblue")
}

# Print descriptive statistics for numerical columns
summary(enron_data[numerical_cols])

# Univariate analysis for categorical columns
categorical_cols <- names(enron_data)[sapply(enron_data, function(col) is.character(col) || is.factor(col))]

# Excluding the 'Unnamed: 0' column (employee names)
categorical_cols <- setdiff(categorical_cols, 'Unnamed: 0')

# Plot bar plots for categorical columns using ggplot2
for (col in categorical_cols) {
  p <- ggplot(enron_data, aes_string(x = col)) +
    geom_bar(fill = "lightblue") +
    labs(title=col) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  print(p)
}



# Bivariate analysis
columns_to_analyze_bivariate <- c('salary', 'total_payments', 'bonus', 'from_poi_to_this_person', 'exercised_stock_options')
plot_list_bivariate <- list()

for (col in columns_to_analyze_bivariate) {
  p <- ggplot(enron_data, aes_string(x=col, y='poi', color='poi')) + 
    geom_point(alpha=0.7) +
    ggtitle(paste(col, 'vs. Person of Interest (POI)')) +
    xlab(col) +
    ylab('POI')
  plot_list_bivariate[[col]] <- p
}

grid.arrange(grobs=plot_list_bivariate, ncol=2)



# Remove columns with more than 50% missing values
enron_data_clean <- enron_data[, colSums(is.na(enron_data)) / nrow(enron_data) < 0.5]

# Compute the correlation matrix for numeric columns using only complete cases
correlation_matrix <- cor(enron_data_clean[sapply(enron_data_clean, is.numeric)], use="complete.obs")

# Plot the correlation matrix
library(corrplot)
corrplot(correlation_matrix, method="color", type="upper", order="hclust", 
         tl.col="black", tl.srt=90)


# Replace NA in numerical columns with 0
numerical_cols <- sapply(enron_data, is.numeric)
enron_data[numerical_cols] <- lapply(enron_data[numerical_cols], function(x) ifelse(is.na(x), 0, x))

# Replace NA in factor or character columns with 'None'
categorical_cols <- sapply(enron_data, function(x) is.character(x) | is.factor(x))
enron_data[categorical_cols] <- lapply(enron_data[categorical_cols], function(x) ifelse(is.na(x), 'None', x))

# Compute the correlation matrix for numeric columns
correlation_matrix <- cor(enron_data[sapply(enron_data, is.numeric)])

# Plot the correlation matrix
corrplot(correlation_matrix, method="color", type="upper", order="hclust", 
         tl.col="black", tl.srt=90)


# Missing Data Analysis
missing_data <- colSums(is.na(enron_data)) / nrow(enron_data) * 100
missing_data <- sort(missing_data, decreasing=TRUE)

# Outlier Analysis using boxplots with ggplot2 for all numeric variables
plots <- list()
numeric_cols <- sapply(enron_data, is.numeric) # Identify numeric columns

for (col in names(enron_data)[numeric_cols]) {
  p <- ggplot(enron_data, aes_string(y = col)) + geom_boxplot() + labs(title=paste("Boxplot of", col))
  plots[[col]] <- p
}

# Display all the plots using grid.arrange
do.call("grid.arrange", c(plots, ncol=3))

print(missing_data)




