library(ggplot2)
library(patchwork)
library(cowplot)

# ============================================================
# Setup: two Beta densities on [0,1]
# ============================================================
x <- seq(0.001, 0.999, length.out = 2000)

# The two densities
p_22  <- dbeta(x, 2, 2)        # Beta(2,2): unimodal, peak at 0.5
p_55  <- dbeta(x, 0.5, 0.5)    # Beta(0.5,0.5): U-shaped, spikes at 0,1

# ============================================================
# Plot 1: The densities themselves
# ============================================================
df_dens <- rbind(
  data.frame(x = x, density = p_22, dist = "Beta(2, 2)"),
  data.frame(x = x, density = p_55, dist = "Beta(0.5, 0.5)")
)

# Cap at 5 for visualization (Beta(0.5, 0.5) shoots to infinity at endpoints)
df_dens$density <- pmin(df_dens$density, 5)

plot_densities <- ggplot(df_dens, aes(x = x, y = density, color = dist)) +
  geom_line(linewidth = 1.1) +
  scale_color_manual(values = c("Beta(2, 2)" = "#1f77b4",
                                "Beta(0.5, 0.5)" = "#d62728")) +
  labs(# title = "Densities",
       # subtitle = "Beta(2,2) is unimodal; Beta(0.5,0.5) is U-shaped (capped at 5)",
       x = "x", y = "Density", color = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = c(0.5, 0.85),
        legend.text = element_text(size = 13),
        axis.title = element_text(size = 17))

# ============================================================
# Plot 2: Decreasing rearrangements
# ============================================================
# The decreasing rearrangement p*(r) is the value t such that
# the level set {p > t} has Lebesgue measure r.
# Equivalently: sort the density values in decreasing order over the support.
# 
# For a density on [0,1] sampled at uniform grid, we can approximate this
# by sorting the density values in decreasing order. The grid spacing
# gives us the "r" coordinate (cumulative volume).

dx <- diff(x)[1]  # grid spacing

# Sort density values in decreasing order
p_22_star <- sort(p_22, decreasing = TRUE)
p_55_star <- sort(p_55, decreasing = TRUE)

# r-coordinate: cumulative volume
r <- seq(dx, 1, length.out = length(x))

df_rearr <- rbind(
  data.frame(r = r, p_star = p_22_star, dist = "Beta(2, 2)"),
  data.frame(r = r, p_star = p_55_star, dist = "Beta(0.5, 0.5)")
)
df_rearr$p_star <- pmin(df_rearr$p_star, 5)  # Cap for visualization

plot_rearr <- ggplot(df_rearr, aes(x = r, y = p_star, color = dist)) +
  geom_line(linewidth = 1.1) +
  scale_color_manual(values = c("Beta(2, 2)" = "#1f77b4",
                                "Beta(0.5, 0.5)" = "#d62728")) +
  labs(# title = "Decreasing Rearrangements",
       # subtitle = "Sort density values from highest to lowest",
       x = "Cumulative volume", y = "Density (sorted)", color = NULL) +
  theme_minimal(base_size = 12) +
  theme(axis.title = element_text(size = 17), legend.position = 'none')

# ============================================================
# Plot 3: Cumulative profiles (the key plot for levelling)
# ============================================================
# P(r) = integral from 0 to r of p*(s) ds
# This is the cumulative integral of the decreasing rearrangement.
# Total mass: P(1) = 1 for any density.
#
# Levelling order: p ≼ p' iff P(r) ≥ P'(r) for all r.
# If the curves cross, neither dominates: incomparable.

P_22 <- cumsum(p_22_star) * dx
P_55 <- cumsum(p_55_star) * dx

df_cumul <- rbind(
  data.frame(r = r, P = P_22, dist = "Beta(2, 2)"),
  data.frame(r = r, P = P_55, dist = "Beta(0.5, 0.5)")
)

# plot_cumul <- ggplot(df_cumul, aes(x = r, y = P, color = dist)) +
#   geom_line(linewidth = 1.1) +
#   scale_color_manual(values = c("Beta(2, 2)" = "#1f77b4",
#                                 "Beta(0.5, 0.5)" = "#d62728")) +
#   labs(title = "Cumulative Profiles",
#        subtitle = "Curves cross — neither distribution levels the other",
#        x = "r", y = "P(r) = ∫₀ʳ p*(s) ds", color = NULL) +
#   theme_minimal(base_size = 12) +
#   theme(legend.position = "top") +
#   geom_hline(yintercept = 1, linetype = "dotted", color = "grey50")


df_cumul$alpha <- 1 - df_cumul$P
plot_coverage <- ggplot(df_cumul, aes(alpha, r, color = dist)) + 
  geom_line(linewidth = 1.1) +
  scale_color_manual(values = c("Beta(2, 2)" = "#1f77b4",
                                "Beta(0.5, 0.5)" = "#d62728")) +
  labs(# title = "Coverage Curves",
       # subtitle = "Curves cross — neither distribution levels the other",
       x = "alpha", y = "alpha-volume", color = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none",
        axis.title = element_text(size = 17)) 




# ============================================================
# Plot 4: Difference of cumulative profiles
# ============================================================
# P_22 - P_55: positive where Beta(2,2) accumulates more mass,
# negative where Beta(0.5,0.5) accumulates more mass.
# If this changed sign, the curves cross and the distributions
# are incomparable.

df_diff <- data.frame(r = r, diff = P_22 - P_55)

plot_diff <- ggplot(df_diff, aes(x = r, y = diff)) +
  geom_line(linewidth = 1.1, color = "darkgreen") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  labs(title = "Difference: P_Beta(2,2) - P_Beta(0.5,0.5)",
       subtitle = "Sign change confirms the cumulative profiles cross",
       x = "r", y = "P(r) difference") +
  theme_minimal(base_size = 12)

# ============================================================
# Combine plots
# ============================================================
combined <- (plot_densities | plot_rearr) / (plot_cumul | plot_diff)

print(combined)

# ============================================================
# Diagnostic: where do the cumulative profiles cross?
# ============================================================
crossing <- which(diff(sign(P_22 - P_55)) != 0)
if (length(crossing) > 0) {
  cat("Cumulative profiles cross at r ≈",
      round(r[crossing], 3), "\n")
  cat("This confirms Beta(2,2) and Beta(0.5,0.5) are incomparable",
      "under the levelling order.\n")
}


### Combine the figures ###
combined_plot <- plot_grid(
  plot_densities, plot_rearr, plot_coverage,
  labels = c("(A)", "(B)", "(C)"),
  nrow = 1,
  align = "hv",
  axis = "tb"
)

combined_plot

ggsave(
  "partial_order.pdf",
  combined_plot,
  width = 16,
  height = 4,
  dpi = 300
)




