library(FactoMineR)
library(factoextra)
library(tidyverse)
library(ggpubr)
library(patchwork)
library(patchwork)
library(cowplot)    

plot_mfa_biplot <- function(mfa_result,
                            axes        = c(1, 2),
                            filter_type = "cos2",
                            filter_value = 0.20,
                            base_size   = 14) {
  
  # ── Selection rule ────────────────────────────────────────────────────────────
  sel <- list();  sel[[filter_type]] <- filter_value
  
  base_sq <- theme_minimal(base_size = base_size) +
    theme(aspect.ratio = 1,
          plot.margin  = margin(2, 2, 2, 2))   # trim panel margins
  
  # ── Make the two core panels (legend removed) ────────────────────────────────
  p_var <- fviz_mfa_var(mfa_result,
                        axes       = axes,
                        select.var = sel,
                        repel      = TRUE,
                        palette    = "Dark2",
                        ggtheme    = base_sq,
                        title      = NULL) +
    coord_fixed()
  
  p_ind <- fviz_mfa_ind(mfa_result,
                        axes      = axes,
                        label     = "none",
                        alpha.ind = 0.40,
                        palette   = "grey40",
                        ggtheme   = base_sq,
                        title     = NULL) +
    coord_fixed()
  
  # ── Extract legend once and place it below both panels ───────────────────────
  legend <- cowplot::get_legend(
    fviz_mfa_var(mfa_result,
                 axes       = axes,
                 select.var = sel,
                 palette    = "Dark2",
                 ggtheme    = theme_minimal(base_size = base_size)) +
      theme(legend.position = "bottom"))
  
  # ── Assemble layout: two squares + centred title + shared legend ─────────────
  core      <- p_var | p_ind                                   # side-by-side
  combined  <- core +
    plot_annotation(
      title = sprintf("MFA Biplot – Dimensions %d and %d",
                      axes[1], axes[2]),
      theme = theme(plot.title = element_text(size  = base_size + 2,
                                              face  = "bold",
                                              hjust = .5))
    )
  
  final_out <- (combined / legend) +        # put legend below
    plot_layout(heights = c(1, .08))  # give it ~8 % of the height
  
  print(final_out)
  invisible(final_out)
}



# Data 
setwd("C:/Users/jrojasa/OneDrive - RAND Corporation/Looking Back to Look Forward/Code")
ema_data <- read.csv("EMA_Output_Expanded.csv", header = TRUE, stringsAsFactors = FALSE)
ema_data <- ema_data %>% select(-X) %>% filter_all(all_vars(is.finite(.))) %>% na.omit()

outcome_vars_original <- c("max_radicalized", "final_radicalized", "wage_cost", 
                           "conservation_cost", "final_resource")
rename_map <- c(
  a_w = "Wage effect on Radicalization", # X
  a_e = "Elite effect on Radicalization", # X
  nat_res_regen = "Natural Resource Regeneration", # X
  delta_extract = "Resource Extraction Rate", # X
  delta = "Rate of radicals converting to moderate", # X
  alpha_w = "Exponent Wage-Resource Link", # X
  conservation_effectiveness = "Conservation Effectiveness", # X
  conservation_unit_cost = "Unit Cost of Conservation", # X
  eta_w = "Wage Target Sensitivity", # L
  eta_a = "Radicalization Threshold Sensitivity", # L
  w_T = "Target Wage", # L
  I_Ta = "Radicalization Threshold", # L
  conservation_investment = "Conservation Investment Effort", # L
  max_radicalized = "Peak Radicalized Population", # M
  final_radicalized = "Final Radicalized Population", # M
  wage_cost = "Cumulative Wage Cost", # M
  conservation_cost = "Total Conservation Cost", # M
  final_resource = "Final Resource Level" # M
)

rename_map_fixed <- setNames(names(rename_map), rename_map)
ema_data <- ema_data %>% rename(!!!rename_map_fixed)

# MFA
mfa_result <- MFA(ema_data,
                  group = c(8, 5, 5),    # 8 uncertainties, 5 levers, 5 metrics
                  type = rep("s", 3),                
                  name.group = c("X", "L", "M"))


# Visualiztions

# Scree plot
fviz_screeplot(mfa_result, 
  addlabels = TRUE,      
  ylim = c(0, 20),         
  ncp = 16,                
  barfill = "steelblue",   
  title = "Scree Plot: Variance Explained by Dimensions"
)

# Cumulative variance plot
eig_values <- get_eigenvalue(mfa_result)

plot(eig_values[,'cumulative.variance.percent'], type="b",
     xlab="Dimension", ylab="Cumulative % of Variance",
     main="Cumulative Variance Explained",
     pch=19)
abline(h=62, col="red", lty=2) # Mark 62% line
abline(v=5, col="blue", lty=2) # Mark 5 dimensions

fviz_mfa_var(mfa_result, "group", 
             repel = TRUE,        
             palette = "jco")   

fviz_contrib(mfa_result, choice = "quanti.var", axes = 1, top = 10,
             fill = "steelblue", title = "Top Variable Contributions to Dimension 1")

# New plots

plot_mfa_biplot(mfa_result,
                axes        = c(2, 3),
                filter_type = "contrib",
                filter_value = 5)
