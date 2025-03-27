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

########################
# Entropy and sparsity #
########################

dir_path <- "/Users/dalejn/PycharmProjects/graphwalk_representation/trained_models_elasticNet_3"

# List all files in the directory that end with .csv
file_names <- list.files(path = dir_path, pattern = "^metrics.*\\.csv$", full.names = TRUE)

# Initialize an empty list to store dataframes
data_list <- list()
results_df <- data.frame(
  model_type = character(),
  hidden_size1 = numeric(),
  bottleneck_size = numeric(),
  beta_value = numeric(),
  representation_sparsity = numeric(),
  representation_entropy = numeric(),
  AC_change = numeric(),
  model_number = numeric(),
  stringsAsFactors = FALSE
)

model_number_instance = 1
# Loop through each file and load it
for (file in file_names) {
  # Extract beta value from filename
  beta_value <- as.numeric(sub(".*_beta_([0-9.]+).*\\.csv$", "\\1", basename(file)))
  
  # Read the CSV file
  temp_df <- read.csv(file)
  temp_df$model_number <- model_number_instance
  # Store the full dataframe in data_list if needed
  data_list[[file]] <- temp_df
  
  # Ensure we have exactly 6 values for AC_change
  AC_change <- numeric(6)
  for (i in 1:6) {
    # Find the cosine similarity for epochs 0 and 95 for each iteration
    epoch_0_sim <- temp_df$cosine_sim[temp_df$epoch == 0][i]
    epoch_95_sim <- temp_df$cosine_sim[temp_df$epoch == 95][i]
    
    # Calculate AC_change
    AC_change[i] <- epoch_0_sim - epoch_95_sim
  }
  
  # Get the first 6 values for each column
  model_type <- temp_df$model_type[1:6]
  hidden_size1 <- temp_df$hidden_size1[1:6]
  bottleneck_size <- temp_df$bottleneck_size[1:6]
  beta <- rep(beta_value, 6)
  representation_sparsity <- subset(temp_df$representation_sparsity, temp_df$epoch==max(temp_df$epoch))
  representation_entropy <- subset(temp_df$representation_entropy, temp_df$epoch==max(temp_df$epoch))
  
  # Create a new dataframe with exactly 6 rows
  new_rows <- data.frame(
    model_type = model_type,
    hidden_size1 = hidden_size1,
    bottleneck_size = bottleneck_size,
    beta_value = beta,
    representation_sparsity = representation_sparsity,
    representation_entropy = representation_entropy,
    AC_change = AC_change,
    model_number = model_number_instance
  )
  
  # Add the new rows to the results dataframe
  results_df <- rbind(results_df, new_rows)
  model_number_instance = model_number_instance + 1
}

# Concatenate all dataframes into one
df <- do.call(rbind, data_list)

df$model_type <- as.factor(df$model_type)

# Add memory categorization to both dataframes
df <- df %>%
  mutate(memory = case_when(
    hidden_size1 == 32 ~ 'medium',
    hidden_size1 == 256 ~ 'high',
    hidden_size1 == 6 ~ 'low',
    TRUE ~ 'unknown'
  ))

results_df <- results_df %>%
  mutate(memory = case_when(
    hidden_size1 == 32 ~ 'medium',
    hidden_size1 == 256 ~ 'high',
    hidden_size1 == 6 ~ 'low',
    TRUE ~ 'unknown'
  ))

df$memory <- as.factor(df$memory)
results_df$memory <- as.factor(results_df$memory)
results_df$model_type <- as.factor(results_df$model_type)
unique(results_df$model_type)

df$sigmoid_C_activation <- sigmoid(df$C_activation)

# sparsity and distributedness over beta values
# 
# # Z-score the representation_entropy and filter
# results_df <- results_df %>%
#   mutate(z_representation_entropy = as.numeric(scale(representation_entropy))) %>% # Convert to numeric
#   filter(z_representation_entropy > -4 & z_representation_entropy < 4) # Keep values within range


# Map model_type to new categories
results_df$model_type <- ifelse(
  results_df$model_type %in% c("hybrid blocked", "pure blocked"), "blocked", "interleaved"
)

# Define a manual color swap using the default R colors
color_values <- c(
  "blocked" = scales::hue_pal()(2)[2],       # Assign interleaved's color to blocked
  "interleaved" = scales::hue_pal()(2)[1]    # Assign blocked's color to interleaved
)

# Plot 1: Representation entropy
ggplot(results_df, aes(x = beta_value, y = scale(representation_entropy), color = model_type)) +
  geom_smooth() +
  scale_color_manual(values = color_values) +  # Use swapped colors
  theme_classic(base_size = 20) +
  labs(
    x = "Beta",
    y = "Representation entropy",
    color = "Schedule"
  ) +
  theme(legend.position = "top")  # Move legend to the top

# Save the plot
ggsave("/Users/dalejn/Desktop/Dropbox/Projects/inProgress/2024-12-navigationSpecialIssue/beta_entropy_square.pdf",
       width = 8, height = 8, units = "in")

cor.test(results_df$beta_value[which(results_df$model_type=="blocked")], scale(results_df$representation_entropy)[which(results_df$model_type=="blocked")])

cor.test(results_df$beta_value[which(results_df$model_type=="interleaved")], scale(results_df$representation_entropy)[which(results_df$model_type=="interleaved")])

cor.test(results_df$beta_value, scale(results_df$representation_entropy))



# Modify results_df for sparsity calculations
results_df <- results_df %>%
  mutate(
    # Calculate reciprocal with a small offset to avoid division by zero
    inverse_representation_sparsity = 1 / (representation_sparsity + 1e-10)
  ) %>%
  # Filter out extreme values before scaling
  filter(
    inverse_representation_sparsity < 50  # Adjust threshold based on data
  ) %>%
  mutate(
    z_inverse_representation_sparsity = as.numeric(scale(inverse_representation_sparsity))  # Normalize
  ) %>%
  filter(
    z_inverse_representation_sparsity > -4 & z_inverse_representation_sparsity < 4  # Z-score range filter
  )

# Plot 2: Representation sparsity
ggplot(results_df, aes(x = beta_value, y = scale(inverse_representation_sparsity), color = model_type)) +
  geom_smooth() +
  scale_color_manual(values = color_values) +  # Use swapped colors
  theme_classic(base_size = 20) +
  labs(
    x = "Beta",
    y = "Representation sparsity",
    color = "Schedule"
  ) +
  theme(legend.position = "top")  # Move legend to the top

# Save the plot
ggsave("/Users/dalejn/Desktop/Dropbox/Projects/inProgress/2024-12-navigationSpecialIssue/beta_sparsity_square.pdf",
       width = 8, height = 8, units = "in")

cor.test(results_df$beta_value[which(results_df$model_type=="blocked")], scale(results_df$inverse_representation_sparsity)[which(results_df$model_type=="blocked")])

cor.test(results_df$beta_value[which(results_df$model_type=="interleaved")], scale(results_df$inverse_representation_sparsity)[which(results_df$model_type=="interleaved")])

cor.test(results_df$beta_value, scale(results_df$inverse_representation_sparsity))


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

########################
# Analyze task results #
########################

task_df <- read.csv('/Users/dalejn/PycharmProjects/graphwalk_representation/model_evaluation_results.csv', header=T)
task_df$target_pair_condition <- paste(task_df$model_schedule, task_df$condition)
task_df$target_pair_condition <- as.factor(task_df$target_pair_condition)
task_df$trial_type <- as.factor(task_df$trial_type)
task_df$model_schedule <- as.factor(task_df$model_schedule)
task_df$condition <- as.factor(task_df$condition)
task_df$memory <-  factor(task_df$memory, levels = c('low', 'medium', 'high'))



### split by beta

# Create new beta categories
final_results$beta_category <- case_when(
  final_results$beta <= 0.1 ~ "Low Beta (0-0.1)",
  final_results$beta >= 0.9 ~ "High Beta (0.9-1)",
  final_results$beta >= 0.4 & final_results$beta <= 0.5 ~ "Mid Beta (0.2-0.8)"
)

# Reorder the factor levels for logical presentation
final_results$beta_category <- factor(final_results$beta_category, 
  levels = c("Low Beta (0-0.1)", "Mid Beta (0.2-0.8)", "High Beta (0.9-1)")
)


# Filter for just Blocked and Interleaved conditions
plot_data <- final_results %>%
  filter(matrix_type %in% c("Blocked -> Integration", "Interleaved -> Integration")) %>%
mutate(beta_category = recode(beta_category, 
                              "Low Beta (0-0.1)" = "0-0.1", 
                              "Mid Beta (0.2-0.8)" = "0.2-0.8", 
                              "High Beta (0.9-1)" = "0.9-1")) %>%
  filter(!is.na(beta_category))

# Ensure the interaction term has the correct levels
plot_data <- plot_data %>%
  mutate(interaction = interaction(matrix_type, beta_category))

levels_interaction <- c("Blocked -> Integration.0-0.1", 
                        "Blocked -> Integration.0.2-0.8", 
                        "Blocked -> Integration.0.9-1", 
                        "Interleaved -> Integration.0-0.1", 
                        "Interleaved -> Integration.0.2-0.8", 
                        "Interleaved -> Integration.0.9-1")

plot_data$interaction <- factor(plot_data$interaction, levels = levels_interaction)

# Define color palette
colors <- c(
  "Blocked -> Integration.0-0.1" = "#BDD7E7",
  "Blocked -> Integration.0.2-0.8" = "#6BAED6",
  "Blocked -> Integration.0.9-1" = "#2171B5",
  "Interleaved -> Integration.0-0.1" = "#FCAE91",
  "Interleaved -> Integration.0.2-0.8" = "#FB6A4A",
  "Interleaved -> Integration.0.9-1" = "#CB181D"
)

# Define labels in the correct order
labels <- c(
  "Blocked -> Integration.0-0.1" = "blocked,\nless sparse and more distributed",
  "Blocked -> Integration.0.2-0.8" = "blocked,\nmix of sparse and distributed",
  "Blocked -> Integration.0.9-1" = "blocked,\nmore sparse and less distributed",
  "Interleaved -> Integration.0-0.1" = "interleaved,\nless sparse and more distributed",
  "Interleaved -> Integration.0.2-0.8" = "interleaved,\nmix of sparse and distributed",
  "Interleaved -> Integration.0.9-1" = "interleaved,\nmore sparse and less distributed"
)

# Create the plot
ggplot(plot_data, aes(x = memory, y = rho, 
                         fill = interaction)) +
  geom_bar(stat = "summary", 
          fun = "mean", 
          position = position_dodge(width = 0.9)) +
  geom_jitter(aes(color = interaction), 
              position = position_jitterdodge(jitter.width=0.1, dodge.width = .9), 
              size = .1, alpha = 0.2) +
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
    values = colors,
    labels = labels
  ) +
  scale_color_manual(
    values = colors,
    guide = "none"
  ) + theme_classic(base_size=20) +
  theme(
    legend.position = "top",
    legend.text = element_text(size=12)
  )

df_z <- plot_data
numeric_cols <- sapply(plot_data, is.numeric)
df_z[numeric_cols] <- scale(plot_data[numeric_cols])

summary(lm(data=df_z, rho ~ matrix_type*beta*memory))
