library(ggplot2)
library(dplyr)
library(tidyverse)
library(lmerTest)

rm(list=ls())

sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

range01 <- function(x){(x-min(x))/(max(x)-min(x))}

mean_ci <- function(x) {
  se <- sd(x, na.rm = TRUE) / sqrt(length(x))
  ci <- qt(0.975, df = length(x) - 1) * se
  data.frame(
    y = mean(x, na.rm = TRUE),
    ymin = mean(x, na.rm = TRUE) - ci,
    ymax = mean(x, na.rm = TRUE) + ci
  )
}

#######
# RSA #
#######

# First matrix (Integration)
matrix1 <- matrix(NA, nrow = 6, ncol = 6)

# Fill in the diagonal values (orange squares)
diag(matrix1) <- 1  # Orange for 1

# Fill in the black squares (using 0.5 for black)
matrix1[1, 2:3] <- 0
matrix1[2, 3] <- 0
matrix1[4, 5:6] <- 0
matrix1[5, 6] <- 0

# Make it symmetric
matrix1[lower.tri(matrix1)] <- t(matrix1)[lower.tri(matrix1)]

# Add row and column names
rownames(matrix1) <- paste0("C", 1:6)
colnames(matrix1) <- paste0("A", 1:6)

# Second matrix (Separation)
matrix2 <- matrix(NA, nrow = 6, ncol = 6)

# Fill in the diagonal values (orange squares)
diag(matrix2) <- -1  # Orange for 1

# Fill in the black squares (using 0.5 for black)
matrix2[1, 2:3] <- 0
matrix2[2, 3] <- 0
matrix2[4, 5:6] <- 0
matrix2[5, 6] <- 0

# Make it symmetric
matrix2[lower.tri(matrix2)] <- t(matrix2)[lower.tri(matrix2)]

# Add row and column names
rownames(matrix2) <- paste0("C", 1:6)
colnames(matrix2) <- paste0("A", 1:6)

# Third matrix (Blocked → Integration)
matrix3 <- matrix(NA, nrow = 6, ncol = 6)

# Fill in the orange squares in top-left quadrant
matrix3[1:3, 1:3] <- diag(3)

# Fill in the light blue squares in bottom-right quadrant (-1 for light blue)
matrix3[4:6, 4:6] <- diag(3) * -1

# Fill in the black squares
matrix3[1, 2:3] <- 0
matrix3[2, 3] <- 0
matrix3[4, 5:6] <- 0
matrix3[5, 6] <- 0

# Make it symmetric
matrix3[lower.tri(matrix3)] <- t(matrix3)[lower.tri(matrix3)]

# Add row and column names
rownames(matrix3) <- paste0("C", 1:6)
colnames(matrix3) <- paste0("A", 1:6)

# Fourth matrix (Interleaved → Integration)
matrix4 <- matrix(NA, nrow = 6, ncol = 6)

# Fill in the orange squares in top-left quadrant
matrix4[1:3, 1:3] <- diag(3) * -1

# Fill in the light blue squares in bottom-right quadrant (-1 for light blue)
matrix4[4:6, 4:6] <- diag(3)

# Fill in the black squares
matrix4[1, 2:3] <- 0
matrix4[2, 3] <- 0
matrix4[4, 5:6] <- 0
matrix4[5, 6] <- 0

# Make it symmetric
matrix4[lower.tri(matrix4)] <- t(matrix4)[lower.tri(matrix4)]

# Add row and column names
rownames(matrix4) <- paste0("C", 1:6)
colnames(matrix4) <- paste0("A", 1:6)


rsa_df <- read.csv('/Users/dalejn/PycharmProjects/graphwalk_representation/RSA_elastic_3.csv', header=T)

# Flatten and name elements of matrix1
flattened_matrix1 <- as.vector(matrix1)
names(flattened_matrix1) <- as.vector(outer(colnames(matrix1), rownames(matrix1), paste, sep = "."))

# Flatten and name elements of matrix2
flattened_matrix2 <- as.vector(matrix2)
names(flattened_matrix2) <- as.vector(outer(colnames(matrix2), rownames(matrix2), paste, sep = "."))

# Flatten and name elements of matrix3
flattened_matrix3 <- as.vector(matrix3)
names(flattened_matrix3) <- as.vector(outer(colnames(matrix3), rownames(matrix3), paste, sep = "."))

# Flatten and name elements of matrix4
flattened_matrix4 <- as.vector(matrix4)
names(flattened_matrix4) <- as.vector(outer(colnames(matrix4), rownames(matrix4), paste, sep = "."))

# Initialize a list to store results
correlation_results <- list()

# Define the correct order of the names as in flattened_matrix1
correct_order <- names(flattened_matrix1)

# Loop through the data
for (i in 1:nrow(rsa_df)) {
  
  # Extract the relevant row (A1.C1, A1.C2, ...)
  row_data <- rsa_df[i, grep("A", names(rsa_df))]
  
  # Count the number of NA values in row_data
  na_count <- sum(is.na(row_data))
  
  # Check if the number of NA values is between 6 and 9
  if (na_count > 30) {
    # If the condition is met, skip this iteration
    next
  }
  
  # Ensure the data is ordered according to the flattened_matrix1 names
  sorted_data <- row_data[match(correct_order, names(row_data))]
  
  # Perform Spearman correlation between the sorted data and each matrix
  result_matrix1 <- cor.test(as.numeric(unlist(sorted_data)), as.numeric(flattened_matrix1), method = "spearman", na.omit = TRUE)
  result_matrix2 <- cor.test(as.numeric(unlist(sorted_data)), as.numeric(flattened_matrix2), method = "spearman", na.omit = TRUE)
  result_matrix3 <- cor.test(as.numeric(unlist(sorted_data)), as.numeric(flattened_matrix3), method = "spearman", na.omit = TRUE)
  result_matrix4 <- cor.test(as.numeric(unlist(sorted_data)), as.numeric(flattened_matrix4), method = "spearman", na.omit = TRUE)
  
  # Extract rho and p-values from each result
  rho1 <- result_matrix1$estimate
  p1 <- result_matrix1$p.value
  
  rho2 <- result_matrix2$estimate
  p2 <- result_matrix2$p.value
  
  rho3 <- result_matrix3$estimate
  p3 <- result_matrix3$p.value
  
  rho4 <- result_matrix4$estimate
  p4 <- result_matrix4$p.value
  
  # Store the results in a tidy format
  correlation_results[[i]] <- data.frame(
    rho = c(rho1, rho2, rho3, rho4),
    p_value = c(p1, p2, p3, p4),
    model_type = rep(rsa_df$model_type[i], 4),
    memory = rep(rsa_df$memory[i], 4),
    beta = rep(rsa_df$beta[i], 4),
    matrix_type = rep(c("matrix1", "matrix2", "matrix3", "matrix4"), each = 1)
  )
}

# Combine all results into a final dataframe
final_results <- do.call(rbind, correlation_results)

# Optional: Reorder columns for better readability
final_results <- final_results[, c("model_type", "memory", "beta", "matrix_type", "rho", "p_value")]


final_results$model_type <- as.factor(final_results$model_type)
final_results$memory <- as.factor(final_results$memory)

# Replace matrix type values with descriptive names
final_results$matrix_type[final_results$matrix_type == "matrix1"] <- "Integration"
final_results$matrix_type[final_results$matrix_type == "matrix2"] <- "Separation"
final_results$matrix_type[final_results$matrix_type == "matrix3"] <- "Blocked -> Integration"
final_results$matrix_type[final_results$matrix_type == "matrix4"] <- "Interleaved -> Integration"
final_results$matrix_type <- as.factor(final_results$matrix_type)

# AC similarity over epochs
final_results$memory <- factor(
  final_results$memory, 
  levels = c("low", "medium", "high"), 
  ordered = TRUE
)

final_results <- final_results %>%
  mutate(model_type = recode(model_type, 
                             "hybrid_repeat" = "hybrid", 
                             "blocked_interleaved" = "pure", 
                             "interleaved_blocked" = "pure"))

final_results <- final_results %>%
  filter(!matrix_type %in% c("Integration", "Separation"))


ggplot(final_results, aes(x = memory, y = rho, fill = matrix_type)) +
  geom_bar(
    stat = "summary", 
    fun = "mean", 
    position = position_dodge(width = 0.9)
  ) +
  geom_jitter(
    aes(x = memory, color = matrix_type), 
    position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.9), 
    size = 0.1, alpha = 0.2
  ) +
  geom_errorbar(
    stat = "summary", 
    fun.data = mean_ci, 
    position = position_dodge(width = 0.9), 
    width = 0.25, 
    color = "black"
  ) +
  labs(
    x = "Memory Condition",
    y = "Correlation",
    fill = ""
  ) +
  scale_fill_manual(
    values = c(
      "Blocked -> Integration" = scales::hue_pal()(2)[2], # Assign interleaved's color to blocked
      "Interleaved -> Integration" = scales::hue_pal()(2)[1] # Assign blocked's color to interleaved
    )
  ) +
  scale_color_manual(
    values = c(
      "Blocked -> Integration" = scales::hue_pal()(2)[2],
      "Interleaved -> Integration" = scales::hue_pal()(2)[1]
    )
  ) +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


df_z <- final_results
numeric_cols <- sapply(final_results, is.numeric)
df_z[numeric_cols] <- scale(final_results[numeric_cols])
summary(lm(data=df_z, rho ~ matrix_type*memory))