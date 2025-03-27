library(ggplot2)
library(dplyr)
library(tidyverse)
library(lmerTest)

rm(list=ls())

sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

range01 <- function(x){(x-min(x))/(max(x)-min(x))}

# Function to calculate 95% confidence interval
mean_ci <- function(x) {
  se <- sd(x, na.rm = TRUE) / sqrt(length(x))
  ci <- qt(0.975, df = length(x) - 1) * se
  data.frame(
    y = mean(x, na.rm = TRUE),
    ymin = mean(x, na.rm = TRUE) - ci,
    ymax = mean(x, na.rm = TRUE) + ci
  )
}


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

# Learning curves

# AC similarity over epochs
df$memory <- factor(
  df$memory, 
  levels = c("low", "medium", "high"), 
  ordered = TRUE
)

# Map model_type to new categories
df$model_type <- ifelse(
  df$model_type %in% c("hybrid blocked", "pure blocked"), "blocked", "interleaved"
)

# Define a manual color swap using the default R colors
color_values <- c(
  "blocked" = scales::hue_pal()(2)[2],       # Assign interleaved's color to blocked
  "interleaved" = scales::hue_pal()(2)[1]    # Assign blocked's color to interleaved
)

ggplot(data = df, aes(x = epoch, y = cosine_sim, group = model_type, fill = model_type)) +
  scale_color_manual(values = color_values) +  # Use swapped colors
  scale_fill_manual(values = color_values) +
  geom_smooth(aes(color=model_type)) +
  facet_wrap(~memory) +
  theme_classic(base_size = 20)


df_z <- df
numeric_cols <- sapply(df, is.numeric)
df_z[numeric_cols] <- scale(df[numeric_cols])

summary(lm(data=df_z, cosine_sim ~ epoch*model_type*memory))

summary(lmer(cosine_sim ~ epoch*model_type*memory + (1 | model_number), data = df_z))

##################
# check training #
##################

# Read the data
df <- read.csv('/Users/dalejn/PycharmProjects/graphwalk_representation/models_performance_elastic_3.csv', header=T)

# Convert relevant columns to factors
df$target_pair_condition <- as.factor(df$target_pair_condition)
df$model_type <- as.factor(df$model_type)
df$memory <- as.factor(df$memory)
df$sigmoid_C_activation <- sigmoid(df$C_activation)

df$memory <- factor(
  df$memory, 
  levels = c("low", "medium", "high"), 
  ordered = TRUE
)

# Check sparsity by memory, split by schedule

df <- df %>%
  mutate(
    # Calculate reciprocal with a small offset to avoid division by zero
    inverse_representation_sparsity = 1 / (representation_sparsity + 1e-10)
  ) %>%
  # Filter out extreme values before scaling
  filter(
    inverse_representation_sparsity < 50  # Adjust this threshold based on your data (or try higher value)
  ) %>%
  mutate(
    z_inverse_representation_sparsity = as.numeric(scale(inverse_representation_sparsity)) # Normalize
  ) %>%
  filter(
    z_inverse_representation_sparsity > -4 & z_inverse_representation_sparsity < 4 # Filter within z-score range
  )

###################
# analyze results #
###################

# AC similarity
# Reorder memory levels
df$memory <- factor(
  df$memory, 
  levels = c("low", "medium", "high"), 
  ordered = TRUE
)
beta=1

# AC similarity over epochs
df$memory <- factor(
  df$memory, 
  levels = c("low", "medium", "high"), 
  ordered = TRUE
)

# Map model_type to new categories
df$target_pair_condition <- ifelse(
  df$target_pair_condition %in% c("hybrid blocked", "pure blocked"), "blocked", "interleaved"
)

# Define a manual color swap using the default R colors
color_values <- c(
  "blocked" = scales::hue_pal()(2)[2],       # Assign interleaved's color to blocked
  "interleaved" = scales::hue_pal()(2)[1]    # Assign blocked's color to interleaved
)

ggplot(df, aes(x = memory, y = cosine_sim, fill = target_pair_condition)) +
  # Bars with means
  stat_summary(fun = mean, geom = "bar", position = "dodge", 
               color = "black", width = 0.7, alpha = 0.7, show.legend = TRUE) +
  
  # Add the 95% Confidence Intervals FIRST
  stat_summary(fun.data = function(x) {
    mean_val <- mean(x)
    se <- sd(x) / sqrt(length(x))
    ci <- qt(0.975, df = length(x)-1) * se  # 95% CI
    return(data.frame(y = mean_val, ymin = mean_val - ci, ymax = mean_val + ci))
  }, geom = "errorbar", position = position_dodge(width = 0.7), width = 0.25) +
  
  geom_jitter(aes(x = memory, color = target_pair_condition), 
              position = position_jitterdodge(jitter.width=0.3, dodge.width = 0.7), size = .1, alpha = 0.3) +
  scale_color_manual(values = color_values) +  # Use swapped colors
  scale_fill_manual(values = color_values) +
  labs(x = "Memory", y = "Cosine Similarity") +
  theme_classic(base_size = 14) +
  theme(legend.position = "top", 
        axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5))

ggsave("/Users/dalejn/Desktop/Dropbox/Projects/inProgress/2024-12-navigationSpecialIssue/AC_integration_bar.pdf",
       width = 12, height = 8, units = "in")

wilcox.test(df$cosine_sim[which(df$target_pair_condition=="blocked" & df$memory=="low")], 
            df$cosine_sim[which(df$target_pair_condition=="interleaved" & df$memory=="low")], )

df_z <- df
numeric_cols <- sapply(df, is.numeric)
df_z[numeric_cols] <- scale(df[numeric_cols])

summary(lm(cosine_sim ~ memory * target_pair_condition, data = df_z))

summary(lmer(cosine_sim ~ memory * target_pair_condition + (1 | model_number), data = df_z))