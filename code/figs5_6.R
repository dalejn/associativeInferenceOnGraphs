rm(list=ls())
library(ggplot2)
library(tidyr)
df = read.csv('/Users/dalejn/Desktop/Dropbox/Projects/inProgress/2024-12-navigationSpecialIssue/rel_dist_dataframe.csv', header=T)
         
str(df)

############
# Figure 5 #
############

mean_ci <- function(x) {
  se <- sd(x, na.rm = TRUE) / sqrt(length(x))
  ci <- qt(0.975, df = length(x) - 1) * se
  data.frame(
    y = mean(x, na.rm = TRUE),
    ymin = mean(x, na.rm = TRUE) - ci,
    ymax = mean(x, na.rm = TRUE) + ci
  )
}

plot_df <- as.data.frame(cbind(df$task, df$L2, df$X1, df$X2, df$X3, df$beta))
colnames(plot_df) <- c('schedule', 'memory', 'D1', 'D2', 'D3', 'beta')

# Pivoting the data longer
plot_df_long <- plot_df %>%
  pivot_longer(cols = starts_with("D"),  # The columns to pivot
               names_to = "distance",     # New column name for the distance
               values_to = "performance") # New column name for the performance scores

# If you want the distance column to be in the correct order (1 to 4)
plot_df_long$distance <- factor(plot_df_long$distance, levels = c("D1", "D2", "D3"))
plot_df_long$schedule <- as.factor(plot_df_long$schedule)
plot_df_long$memory <- as.numeric(plot_df_long$memory)
plot_df_long$performance <- as.numeric(plot_df_long$performance)
plot_df_long$beta <- as.numeric(plot_df_long$beta)

# View the result
head(plot_df_long)

# Factorize the distance column to order the facets correctly
plot_df_long$distance <- factor(plot_df_long$distance, levels = c("D1", "D2", "D3"))

# Define a manual color swap using the default R colors
color_values <- c(
  "blocked" = scales::hue_pal()(2)[2],       # Assign interleaved's color to blocked
  "intermixed" = scales::hue_pal()(2)[1]    # Assign blocked's color to interleaved
)

beta=0
# Creating the plot with log2 scale on x-axis and classic theme
ggplot(plot_df_long[which(plot_df_long$beta %in% c(0, 0.1, 0.2, 0.3)), ], aes(x = memory, y = performance, color = schedule)) +
  # geom_point() +  # Scatter plot of performance vs memory
  geom_jitter(aes(color = schedule), 
            position = position_jitterdodge(jitter.width = 0.4, dodge.width = 0.2),
            size = 1, 
            alpha = 0.2) +
  geom_smooth(method = "lm", se = TRUE) +  # Add a linear regression line
  facet_wrap(~ distance, scales = "free_y") +  # Create a facet plot for each distance
  labs(title = "Performance vs Memory by Schedule and Distance",
       x = "Memory (L2)",
       y = "Performance",
       color = "Schedule") +
  geom_hline(yintercept=50, linetype='dotted')+
  scale_fill_manual(values=color_values) +
  scale_color_manual(values=color_values) +
  theme_classic() +  # Apply the classic theme
  scale_x_continuous(trans = 'log2') +  # Apply log2 scale to x-axis
  theme(legend.position = "top")+  # Position the legend at the top
  ylim(0,100)

ggsave("/Users/dalejn/Desktop/Dropbox/Projects/inProgress/2024-12-navigationSpecialIssue/graphwalk_performance_by_distance.pdf",
       width = 12, height = 6, units = "in")

plot_df_long$distance <- factor(plot_df_long$distance, levels = c("D1", "D2", "D3"), ordered = TRUE)

df_z <- plot_df_long
df_z$memory <- scale(df_z$memory)
df_z$performance <- scale(df_z$performance)

summary(lm(data=df_z[which(plot_df_long$beta %in% c(0, 0.1, 0.2, 0.3)), ], performance~schedule*memory*distance))

###############################
# Figure 6A Check integration #
###############################

plot_df <- as.data.frame(cbind(df$task, df$L2, df$beta, df$integration))
colnames(plot_df) <- c('schedule', 'memory', 'beta', 'integration')

plot_df$schedule <- as.factor(plot_df$schedule)
plot_df$memory <- as.numeric(plot_df$memory)
plot_df$beta <- as.numeric(plot_df$beta)
plot_df$integration <- as.numeric(plot_df$integration)

ggplot(plot_df[which(plot_df$beta %in% c(0.1, 0.2, 0.3, 0.4)),], aes(x = memory, y = integration, color = schedule)) +
  geom_jitter(aes(color = schedule), 
            position = position_jitterdodge(jitter.width = 0.4, dodge.width = 0.2),
            size = 1, 
            alpha = 0.2) +
  geom_smooth(method = "lm") +  # Add a linear regression line
  facet_wrap(~ beta, scales = "free_y", nrow=1, ncol=4) +  # Create a facet plot for each distance
  labs(title = "Integration vs Memory by Schedule and Distance",
       x = "Memory (L2)",
       y = "Integration",
       color = "Schedule") +
  theme_classic() +  # Apply the classic theme
  scale_fill_manual(values=color_values) +
  scale_color_manual(values=color_values) +
  scale_x_continuous(trans = 'log2') +  # Apply log2 scale to x-axis
  theme(legend.position = "top") +   # Position the legend at the top
  geom_hline(yintercept=0, linetype='dotted')+
  ylim(c(-.1, 0.65))
  
ggsave("/Users/dalejn/Desktop/Dropbox/Projects/inProgress/2024-12-navigationSpecialIssue/graphwalk_performance_by_beta.pdf",
       width = 12, height = 6, units = "in")


df_z <- plot_df
df_z$memory <- scale(df_z$memory)
df_z$integration <- scale(df_z$integration)

summary(lm(data=df_z[which(df_z$beta %in% c(0.1, 0.2, 0.3, 0.4)),], integration~schedule*memory*beta))

plot_df$beta_category <- NA  # Start with all NA
plot_df$beta_category[plot_df$beta >= 0 & plot_df$beta <= 0.2] <- "Low"
plot_df$beta_category[plot_df$beta >= 0.3 & plot_df$beta <= 0.7] <- "Medium"
plot_df$beta_category[plot_df$beta >= 0.8 & plot_df$beta <= 1] <- "High"
plot_df$beta_category <- factor(plot_df$beta_category, 
                              levels = c("Low", "Medium", "High"))

# Calculate the 33rd and 67th percentiles
memory_thirds <- quantile(plot_df$memory, probs = c(1/3, 2/3))

# Create categories based on these thirds
plot_df$memory_category <- cut(plot_df$memory,
                             breaks = c(-Inf, memory_thirds[1], memory_thirds[2], Inf),
                             labels = c("Low", "Medium", "High"))

# Convert to factor to ensure proper ordering
plot_df$memory_category <- factor(plot_df$memory_category, 
                                levels = c("Low", "Medium", "High"))

plot_df <- plot_df[!is.na(plot_df$beta_category), ]

plot_df$schedule_beta <- interaction(plot_df$schedule, plot_df$beta_category)
plot_df$schedule_beta <- factor(plot_df$schedule_beta, 
                                   levels = c("blocked.Low",
                                            "blocked.Medium",
                                            "blocked.High",
                                            "intermixed.Low",
                                            "intermixed.Medium",
                                            "intermixed.High"))

# Color palette
colors <- c(
  "blocked.Low" = "#BDD7E7",
  "blocked.Medium" = "#6BAED6",
  "blocked.High" = "#2171B5",
  "intermixed.Low" = "#FCAE91",
  "intermixed.Medium" = "#FB6A4A",
  "intermixed.High" = "#CB181D"
)

# Create the plot with ordered bars
ggplot(plot_df, aes(x = memory_category, y = integration, 
                         fill = schedule_beta)) +  # Use the new ordered factor
  geom_bar(stat = "summary", 
           fun = "mean", 
           position = position_dodge(width = 0.9)) +
  geom_jitter(aes(color = schedule_beta), 
              position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.9),
              size = 0.1, 
              alpha = 0.2) +
  geom_errorbar(stat = "summary",
                fun.data = mean_ci,
                position = position_dodge(width = 0.9),
                width = 0.25,
                color = "black") +
  labs(x = "Memory",
       y = "Integration",
       fill = "") +
  scale_fill_manual(values = colors,
                    labels = function(x) gsub("\\.", ",\n", x)) +
  scale_color_manual(values = colors,
                     guide = "none") +
  theme_classic(base_size = 20) +
  theme(legend.position = "top",
        legend.text = element_text(size = 12))

############ integration by distance

plot_df <- as.data.frame(cbind(df$task, df$L2, df$beta, df$integration_1, df$integration_2, df$integration_3))
colnames(plot_df) <- c('schedule', 'memory', 'beta', 'integration_1', 'integration_2', 'integration_3')

plot_df$schedule <- as.factor(plot_df$schedule)
plot_df$memory <- as.numeric(plot_df$memory)
plot_df$beta <- as.numeric(plot_df$beta)
plot_df$integration_1 <- as.numeric(plot_df$integration_1)
plot_df$integration_2 <- as.numeric(plot_df$integration_2)
plot_df$integration_3 <- as.numeric(plot_df$integration_3)


plot_df$beta_category <- NA  # Start with all NA
plot_df$beta_category[plot_df$beta >= 0 & plot_df$beta <= 0.2] <- "Low"
plot_df$beta_category[plot_df$beta >= 0.3 & plot_df$beta <= 0.7] <- "Medium"
plot_df$beta_category[plot_df$beta >= 0.8 & plot_df$beta <= 1] <- "High"
plot_df$beta_category <- factor(plot_df$beta_category, 
                              levels = c("Low", "Medium", "High"))

# Calculate the 33rd and 67th percentiles
memory_thirds <- quantile(plot_df$memory, probs = c(1/3, 2/3))

# Create categories based on these thirds
plot_df$memory_category <- cut(plot_df$memory,
                             breaks = c(-Inf, memory_thirds[1], memory_thirds[2], Inf),
                             labels = c("Low", "Medium", "High"))

# Convert to factor to ensure proper ordering
plot_df$memory_category <- factor(plot_df$memory_category, 
                                levels = c("Low", "Medium", "High"))

# Pivoting the data longer
plot_df_long <- plot_df %>%
  pivot_longer(cols = starts_with("integration"),  # The columns to pivot
               names_to = "integration_distance",     # New column name for the distance
               values_to = "integration") # New column name for the performance scores

# If you want the distance column to be in the correct order (1 to 4)
plot_df_long$integration_distance <- factor(plot_df_long$integration_distance, levels = c("integration_1", "integration_2", "integration_3"))
plot_df_long$schedule <- as.factor(plot_df_long$schedule)
plot_df_long$memory <- as.numeric(plot_df_long$memory)
plot_df_long$integration <- as.numeric(plot_df_long$integration)
plot_df_long$beta <- as.numeric(plot_df_long$beta)

# View the result
head(plot_df_long)

plot_df_long$schedule_beta <- interaction(plot_df_long$schedule, plot_df_long$beta_category)
plot_df_long$schedule_beta <- factor(plot_df_long$schedule_beta, 
                                   levels = c("blocked.Low",
                                            "blocked.Medium",
                                            "blocked.High",
                                            "intermixed.Low",
                                            "intermixed.Medium",
                                            "intermixed.High"))

# Color palette
colors <- c(
  "blocked.Low" = "#BDD7E7",
  "blocked.Medium" = "#6BAED6",
  "blocked.High" = "#2171B5",
  "intermixed.Low" = "#FCAE91",
  "intermixed.Medium" = "#FB6A4A",
  "intermixed.High" = "#CB181D"
)

# Create the plot with ordered bars
ggplot(plot_df_long, aes(x = memory_category, y = integration, 
                         fill = schedule_beta)) +  # Use the new ordered factor
  geom_bar(stat = "summary", 
           fun = "mean", 
           position = position_dodge(width = 0.9)) +
  geom_jitter(aes(color = schedule_beta), 
              position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.9),
              size = 0.1, 
              alpha = 0.2) +
  geom_errorbar(stat = "summary",
                fun.data = mean_ci,
                position = position_dodge(width = 0.9),
                width = 0.25,
                color = "black") +
  facet_wrap(~ integration_distance, scales = "free_y") +
  labs(x = "Memory",
       y = "Integration",
       fill = "") +
  scale_fill_manual(values = colors,
                    labels = function(x) gsub("\\.", ",\n", x)) +
  scale_color_manual(values = colors,
                     guide = "none") +
  theme_classic(base_size = 20) +
  theme(legend.position = "top",
        legend.text = element_text(size = 12))+
  ylim(c(-.5,1))

ggsave("/Users/dalejn/Desktop/Dropbox/Projects/inProgress/2024-12-navigationSpecialIssue/graphwalk_integration_allBetas.pdf",
       width = 12, height = 8, units = "in")

plot_df_long$integration_distance <- factor(
  plot_df_long$integration_distance,
  levels = c("integration_1", "integration_2", "integration_3"),
  ordered = TRUE
)

plot_df_long$memory_category <- factor(plot_df_long$memory_category, levels = c("Low", "Medium", "High"), , ordered = TRUE)
plot_df_long$schedule_beta <- factor(
  plot_df_long$schedule_beta,
  levels = c("blocked.Low", "blocked.Medium", "blocked.High",
             "intermixed.Low", "intermixed.Medium", "intermixed.High"),
  ordered = TRUE
)

df_z <- plot_df_long
df_z$memory <- scale(df_z$memory)
df_z$integration <- scale(df_z$integration)

summary(lm(data=df_z, integration ~ schedule*memory_category*beta_category + integration_distance))
