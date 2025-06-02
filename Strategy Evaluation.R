library(tidyverse)
library(randomForest)
library(janitor)
library(FactoMineR)
library(factoextra)    
library(dendextend)    
library(prim)          
library(mco)            
library(patchwork)
library(clipr)

# setwd("C:/Users/jrojasa/OneDrive - RAND Corporation/Looking Back to Look Forward/Code")
setwd("/Users/javier/lblf")

# Renaming
rename_map <- c(
  a_w = "Wage → Radicalization",
  a_e = "Elite → Radicalization",
  nat_res_regen = "Resource Regeneration",
  delta_extract = "Extraction Rate",
  delta = "De-radicalization Rate",
  alpha_w = "Wage–Resource Exp.",
  conservation_effectiveness = "Conservation Effectiveness",
  conservation_unit_cost  = "Conservation Unit Cost",
  eta_deplet = "Wage→Depletion Exp.",
  mu_elite_extr = "Elite Extraction Mult.",
  mu_0 = "Elite Mobility Rate",
  e_0 = "Elite Proportion",
  eta_w = "Wage Target Sensitivity",
  eta_a = "Radical. Treshold Sensitivity",
  w_T = "Target Wage",
  I_Ta = "Radical Threshold",
  conservation_effort = "Conservation Effort",
  max_radicalized = "Peak Radicalization",
  final_radicalized = "Final Radicalization",
  wage_cost = "Wage Cost",
  conservation_cost = "Conservation Cost",
  final_resource = "Final Resource"
  #total_policy_cost = "Total Policy Cost"
)

#### Stay the Course
stay_df <- read.csv("EMA_Output_BaseRun.csv")
# stay_df <- stay_df %>% select(-X)
stay_df <- stay_df %>% rename(!!!set_names(names(rename_map), rename_map))
stay_df <- stay_df %>%
  mutate(rad_bin = cut(`Peak Radicalization`,
                       breaks = seq(0, 1, by = 0.05),
                       include.lowest = TRUE,
                       right = FALSE))
# In 5% Buckets
summary_by_bin <- stay_df %>%
  group_by(rad_bin) %>%
  summarise(across(where(is.numeric),  \(x) mean(x, na.rm = TRUE)), .groups = "drop")

print(summary_by_bin, n = nrow(summary_by_bin), width = Inf)

stay_df %>% filter(`Final Resource` > 0) %>% 
  count(rad_bin) %>%
  mutate(prop = round(100 * n / sum(n), 2)) %>%
  write_clip()

# In Cumulative 5% Buckets
max_val <- ceiling(max(stay_df$`Peak Radicalization`, na.rm = TRUE))
bin_edges <- seq(0.05, max_val, by = 0.05)

summary_by_cumbin <- map_dfr(bin_edges, function(upper) {
  stay_df %>%
    filter(`Peak Radicalization` <= upper) %>%
    summarise(across(where(is.numeric), mean, na.rm = TRUE)) %>%
    mutate(rad_bin = paste0("0–", upper)) %>% 
    select(rad_bin, everything())
})

# Plots

vars_to_plot <- c("Elite → Radicalization",
                  "Wage → Radicalization",
                  "Final Resource",
                  "Final Radicalization",
                  "Resource Regeneration")

summary_by_cumbin %>%
  select(rad_bin, all_of(vars_to_plot)) %>%
  pivot_longer(cols = -rad_bin, names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = rad_bin, y = Value, group = Variable)) +
  geom_line(size = 1, color = "steelblue") +
  facet_wrap(~Variable, scales = "free_y") +
  labs(title = "Key System Trends Across Cumulative Radicalization Bins",
       x = "Cumulative Peak Radicalization Bin",
       y = "Average Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(face = "bold"))

summary_by_bin %>%
  select(rad_bin, all_of(vars_to_plot)) %>%
  pivot_longer(cols = -rad_bin, names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = rad_bin, y = Value, group = Variable)) +
  geom_line(size = 1, color = "darkred") +
  facet_wrap(~Variable, scales = "free_y") +
  labs(title = "System Profiles by Exclusive Peak Radicalization Bins",
       x = "Peak Radicalization Bin",
       y = "Average Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(face = "bold"))

# Studying Low Radicalization Cases via PRIM

uncertainty_vars <- c(
  "Wage → Radicalization", "Elite → Radicalization", "Resource Regeneration",
  "Extraction Rate", "De-radicalization Rate", "Wage–Resource Exp.",
  "Conservation Effectiveness", "Conservation Unit Cost",
  "Wage→Depletion Exp.", "Elite Extraction Mult.", 
  "Elite Mobility Rate", "Elite Proportion"
)

stay_df_clean <- stay_df %>%
  filter(!if_any(all_of(uncertainty_vars), is.na)) %>%
  select(all_of(c(uncertainty_vars, "Peak Radicalization")))

X <- stay_df_clean %>%
  select(all_of(uncertainty_vars))

success_vec <- stay_df_clean$`Peak Radicalization` < 0.10

thresholds <- seq(0.05, 1.0, by = 0.05)

prim_box <- prim(X, success_vec, threshold = 0.8)

#### Moderate Response

# Renaming
rename_map <- c(
  a_w = "Wage → Radicalization",
  a_e = "Elite → Radicalization",
  nat_res_regen = "Resource Regeneration",
  delta_extract = "Extraction Rate",
  delta = "De-radicalization Rate",
  alpha_w = "Wage–Resource Exp.",
  conservation_effectiveness = "Conservation Effectiveness",
  conservation_unit_cost  = "Conservation Unit Cost",
  eta_deplet = "Wage→Depletion Exp.",
  mu_elite_extr = "Elite Extraction Mult.",
  mu_0 = "Elite Mobility Rate",
  e_0 = "Elite Proportion",
  eta_w = "Wage Target Sensitivity",
  eta_a = "Radical. Treshold Sensitivity",
  w_T = "Target Wage",
  I_Ta = "Radical Threshold",
  conservation_effort = "Conservation Effort",
  max_radicalized = "Peak Radicalization",
  final_radicalized = "Final Radicalization",
  wage_cost = "Wage Cost",
  conservation_cost = "Conservation Cost",
  final_resource = "Final Resource",
  total_policy_cost = "Total Policy Cost"
)

moderate_df <- read.csv("EMA_Output_LeverRun.csv")
# moderate_df <- moderate_df %>% select(-X)
moderate_df <- moderate_df %>% mutate(total_policy_cost = wage_cost + conservation_cost)
moderate_df <- moderate_df %>% rename(!!!set_names(names(rename_map), rename_map))
moderate_df <- moderate_df %>% filter(`Final Resource` >= 0)
moderate_df <- moderate_df %>%
  mutate(rad_bin = cut(`Peak Radicalization`,
                       breaks = seq(0, 1, by = 0.05),
                       include.lowest = TRUE,
                       right = FALSE))

moderate_df %>%
  group_by(rad_bin) %>%
  summarise(n = n(), .groups = "drop") %>%
  mutate(share = round(n / sum(n), 3))

# Plots

ggplot(moderate_df, aes(x = `Total Policy Cost`, y = `Peak Radicalization`, color = `Conservation Effort`)) +
  geom_jitter(alpha = 0.7) +
  scale_color_viridis_c() +
  theme_minimal() +
  labs(title = "Radicalization vs Policy Cost",
       subtitle = "Color: Conservation Effort",
       x = "Total Policy Cost",
       y = "Peak Radicalization",
       color = "Conservation\nEffort")

ggplot(moderate_df, aes(x = `Total Policy Cost`, y = `Peak Radicalization`, color = `Radical Threshold`)) +
  geom_jitter(alpha = 0.7) +
  scale_color_viridis_c() +
  theme_minimal() +
  labs(title = "Radicalization vs Policy Cost",
       subtitle = "Color: Radical. Treshold",
       x = "Total Policy Cost",
       y = "Peak Radicalization",
       color = "Radicalization \n Treshold")

ggplot(moderate_df, aes(x = `Total Policy Cost`, y = `Peak Radicalization`, color = `Target Wage`)) +
  geom_jitter(alpha = 0.7) +
  scale_color_viridis_c() +
  theme_minimal() +
  labs(title = "Radicalization vs Policy Cost",
       subtitle = "Color: Target Wage",
       x = "Total Policy Cost",
       y = "Peak Radicalization",
       color = "Target Wage")

ggplot(moderate_df, aes(x = `Total Policy Cost`, y = `Peak Radicalization`, color = `Radical. Treshold Sensitivity`)) +
  geom_jitter(alpha = 0.7) +
  scale_color_viridis_c() +
  theme_minimal() +
  labs(title = "Radicalization vs Policy Cost",
       subtitle = "Color: Radical. Treshold Sensitivity",
       x = "Total Policy Cost",
       y = "Peak Radicalization",
       color = "Radical. Treshold \n Sensitivity")


ggplot(moderate_df, aes(x = `Total Policy Cost`, y = `Peak Radicalization`, color = `Wage Target Sensitivity`)) +
  geom_jitter(alpha = 0.7) +
  scale_color_viridis_c() +
  theme_minimal() +
  labs(title = "Radicalization vs Policy Cost",
       subtitle = "Color: Wage Target Sensitivity",
       x = "Total Policy Cost",
       y = "Peak Radicalization",
       color = "Wage Target Sensitivity")

# Clustering Success Cases

low_rad_df <- moderate_df %>%
  arrange(`Peak Radicalization`) %>%
  slice_head(n = 10000) %>%
  select(-rad_bin)

pca_res <- PCA(low_rad_df)
pca_coords <- pca_res$ind$coord[, 1:10]
hc <- hclust(dist(pca_coords), method = "ward.D2")
dend <- as.dendrogram(hc) # 4 groups

plot(dend,
     main = "Dendrogram of PCA-Reduced Low-Radicalization Runs",
     ylab = "Height",
     leaflab = "none")

# Dendogram

dend <- color_branches(dend, k = 4)

plot(dend,
     main = "Dendrogram of PCA-Reduced Low-Radicalization Runs",
     ylab = "Height",
     leaflab = "none")

low_rad_df$cluster <- cutree(hc, k = 4)

# K Means

pca_df <- as.data.frame(pca_coords)






levers_summary <- low_rad_df %>%
  group_by(cluster) %>%
  summarise(across(c(`Conservation Effort`, `Radical Threshold`,
                     `Wage Target Sensitivity`, `Radical. Treshold Sensitivity`),
                   mean, .names = "avg_{.col}"))


uncertainty_vars <- c(
  "Wage → Radicalization", "Elite → Radicalization", "Resource Regeneration",
  "Extraction Rate", "De-radicalization Rate", "Wage–Resource Exp."
)

uncertainties_summary <- low_rad_df %>%
  group_by(cluster) %>%
  summarise(across(all_of(uncertainty_vars), mean, .names = "avg_{.col}"))