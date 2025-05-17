library(tidyverse)      
library(janitor) 
library(FactoMineR) 
library(factoextra)     
library(dendextend)     
library(prim)           
library(mco)            
library(patchwork)      

setwd("C:/Users/jrojasa/OneDrive - RAND Corporation/Looking Back to Look Forward/Code")
# Read and clean data
df <- read.csv("EMA_Output_LeverRun.csv")
df <- df[, -1]

df <- df %>% filter(final_resource > 0) %>% drop_na()

# df %>% write_csv("EMA_Output_Expanded_RSet.csv")

# Scatter plot

df <- df %>%
  mutate(total_policy_cost = wage_cost + conservation_cost)

ggplot(df, aes(x = conservation_cost, y = max_radicalized, fill = conservation_effort)) +
  geom_point(alpha = 0.6) +
  labs(
    title = "Peak Radicalization vs. Conservation Cost",
    x = "Total Policy Cost (Wage + Conservation)",
    y = "Peak Radicalization"
  ) +
  theme_minimal(base_size = 13)


# Renaming
rename_map <- c(
  a_w                         = "Wage → Radicalization",
  a_e                         = "Elite → Radicalization",
  nat_res_regen              = "Resource Regeneration",
  delta_extract              = "Extraction Rate",
  delta                      = "De-radicalization Rate",
  alpha_w                    = "Wage–Resource Exp.",
  conservation_effectiveness = "Conservation Effectiveness",
  conservation_unit_cost     = "Conservation Unit Cost",
  eta_deplet                 = "Wage→Depletion Exp.",
  mu_elite_extr              = "Elite Extraction Mult.",
  mu_0                       = "Elite Mobility Rate",
  e_0                        = "Elite Proportion",
  eta_w                      = "Wage Target Sensitivity",
  eta_a                      = "Elite Treshold Sensitivity",
  w_T                        = "Target Wage",
  I_Ta                       = "Radical Threshold",
  conservation_effort        = "Conservation Effort",
  max_radicalized            = "Peak Radicalization",
  final_radicalized          = "Final Radicalization",
  wage_cost                  = "Wage Cost",
  conservation_cost          = "Conservation Cost",
  final_resource             = "Final Resource",
  total_policy_cost          = "Total Policy Cost"
)

df <- df %>% rename(!!!set_names(names(rename_map), rename_map))

### PCA

pca_res <- PCA(df, scale.unit = TRUE, ncp = 8)

for (ax in list(c(1,2), c(3,4), c(5,6))) {
  print(
    fviz_pca_var(
      pca_res, axes = ax,
      select.var = list(cos2 = 0.1),   # keeps only strong contributors
      repel = TRUE
    ) +
      ggtitle(glue::glue("Correlation circle – (Dim {ax[1]}, Dim {ax[2]})"))
  )
}

### Clusters

pc_scores <- pca_res$ind$coord[, 1:6] %>%
  as_tibble(.name_repair = ~ paste0("PC", seq_along(.x)))

hc <- pc_scores %>%
  dist() %>%
  hclust(method = "ward.D2")


df_pca <- bind_cols(pc_scores, df)
df_pca <- df_pca %>% mutate(cluster = cutree(hc, k = 6))

df_pca %>% group_by(cluster) %>% summarise(n = n())

df_pca %>%
  group_by(cluster) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE), .groups = "drop") %>% 
  c()

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Worst-outcome cluster & PRIM scenario discovery -------------------------
# ──────────────────────────────────────────────────────────────────────────────
worst_cluster <- df_pca %>% 
  group_by(cluster) %>% 
  summarise(mean_pr = mean(`Peak Radicalized Population`, na.rm = TRUE)) %>% 
  slice_max(mean_pr, n = 1) %>% 
  pull(cluster)

df_worst <- df_pca %>% filter(cluster == worst_cluster)

prim_box <- prim(
  X = df_worst %>% select(all_of(lever_cols)),
  y = df_worst$`Peak Radicalized Population`,
  threshold = quantile(df$`Peak Radicalized Population`, 0.90)
)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Policy ranking inside worst cluster  ------------------------------------
# ──────────────────────────────────────────────────────────────────────────────
policy_rank <- df_worst %>% 
  group_by(policy_name) %>%              # adjust if column differs
  summarise(across(
    c(`Peak Radicalized Population`, `Total Policy Cost`), median, na.rm = TRUE),
    .groups = "drop"
  ) %>% 
  arrange(`Peak Radicalized Population`)

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Pareto front (PR vs. Total Cost)  ---------------------------------------
# ──────────────────────────────────────────────────────────────────────────────
pareto_idx <- pareto.filter(
  df %>% select(`Peak Radicalized Population`, `Total Policy Cost`)
)$pareto.optimal

df <- df %>% 
  mutate(pareto = if_else(row_number() %in% pareto_idx, "front", "dominated"))

# ──────────────────────────────────────────────────────────────────────────────
# 7.  Visuals (FactoMineR + tidyverse)  ---------------------------------------
# ──────────────────────────────────────────────────────────────────────────────
gg_pca <- fviz_pca_biplot(
  pca_res, repel = TRUE, label = "var"
) +
  geom_point(
    data = df_pca,
    aes(PC1, PC2, colour = factor(cluster)),
    alpha = 0.5
  ) +
  labs(colour = "Cluster")

gg_dend <- as.dendrogram(hc) %>% 
  set("branches_k_color", k = k) %>% 
  fviz_dend(show_labels = FALSE, main = "Lever-space dendrogram")

gg_pareto <- df %>% 
  ggplot(aes(`Total Policy Cost`, `Peak Radicalized Population`)) +
  geom_point(aes(colour = pareto, alpha = pareto)) +
  scale_colour_manual(values = c(front = "red", dominated = "grey60"), guide = "none") +
  scale_alpha_manual(values  = c(front = 1,    dominated = 0.25),    guide = "none") +
  labs(x = "Total Policy Cost (Wage + Conservation)",
       y = "Peak Radicalized Population")

(combine <- (gg_pca / gg_dend / gg_pareto) +
    plot_layout(heights = c(2, 1.2, 2)))

ggsave("classification_and_tradeoffs.png", combine, width = 9, height = 13)