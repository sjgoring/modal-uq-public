# Load libraries
library(ggplot2)
library(ggsci)
library(cowplot)
library(dplyr)
library(tidyr)
library(patchwork)

### Splash figure ###

# Target: fix the HDR to cover 90% HDR to [-1.645, 1.645] (the std normal HDR)
# and progressively increase variance.

# Helper: given a density function on a grid, return a make_panel tibble
mass <- 2 * pnorm(1) - 1   # ≈ 0.6827

make_panel <- function(x_grid, density, distribution, mass = 2 * pnorm(1) - 1) {
  dx <- diff(x_grid)[1]
  density <- density / sum(density * dx)
  
  ord      <- order(density, decreasing = TRUE)
  cum_mass <- cumsum(density[ord] * dx)
  cutoff   <- density[ord][which(cum_mass >= mass)[1]]
  hdr      <- density >= cutoff
  
  mu    <- sum(x_grid * density * dx)
  sigma <- sqrt(sum((x_grid - mu)^2 * density * dx))
  
  tibble(
    x                = x_grid,
    density          = density,
    distribution     = distribution,
    in_hdr           = hdr,
    in_mean_interval = x_grid >= mu - sigma & x_grid <= mu + sigma
  )
}

# ── Panel 1: Standard normal ──────────────────────────────────────────────────
x1  <- seq(-4.5, 4.5, length.out = 4000)
d1  <- dnorm(x1, 0, 1)

# ── Panel 2: Skewed  -- gamma distro  ──────────────────────────
# x2     <- seq(0, qlnorm(0.99, meanlog = 1, sdlog = 0.75), length.out = 8000)
# d2_raw <- dlnorm(x2, meanlog = 1, sdlog = 0.75)
x2 <- seq(0, qgamma(0.99, shape = 1.15, rate = 1), length.out = 8000)
d2 <- dgamma(x2, shape = 1.15, rate = 1)

# ── Panel 3: Heavy-tailed mix — std normal + far outlier component ────────────
# Mix: w * N(0,1) + (1-w) * N(mu_out, 1), tune mu_out for large variance
# while keeping HDR ≈ the normal HDR (the outlier component is low-density
# everywhere in the HDR region).
w       <- 0.45
eps     <- 0.3
mu_out  <- 20       # far enough that it doesn't intrude on the HDR
x3      <- seq(-4.5, 25, length.out = 8000)
d3      <- w * dnorm(x3, 0, 1) + (1 - eps - w) * dnorm(x3, mu_out, 1.5) + 
  eps * dunif(x3, x3[1], x3[length(x3)])

# ── Build data frame ──────────────────────────────────────────────────────────
plot_df <- bind_rows(
  make_panel(x1, d1, "Normal"),
  make_panel(x2, d2, "Skewed"),  
  make_panel(x3, d3, "Bimodal")
) %>%
  mutate(
    distribution = factor(
      distribution,
      levels = c("Normal", "Skewed", "Bimodal")
    )
  )

# ── Shade data ────────────────────────────────────────────────────────────────
shade_df <- plot_df %>%
  pivot_longer(
    cols      = c(in_hdr, in_mean_interval),
    names_to  = "region",
    values_to = "included"
  ) %>%
  mutate(
    region = recode(region,
                    in_hdr           = "HDR",
                    in_mean_interval = "µ ± σ"
    ),
    region = factor(region, levels = c("HDR", "µ ± σ"))
  ) %>%
  # Assign contiguous-run IDs within each distribution × region combination
  group_by(distribution, region) %>%
  mutate(
    run_id = cumsum(included != lag(included, default = first(included))),
    group  = paste(distribution, region, run_id, sep = "_")
  ) %>%
  ungroup() %>%
  filter(included)

p_hdr <- ggplot(plot_df, aes(x = x, y = density)) +
  geom_ribbon(
    data = shade_df,
    aes(ymin = 0, ymax = density, fill = region, group = group),
    alpha = 0.4
  ) +
  geom_line(linewidth = 0.9, colour = "grey20") +
  facet_wrap(~ distribution, nrow = 1, scales = "free") +
  scale_fill_manual(
    values = c(
      "HDR"            = "#D62839",
      "µ ± σ"       = "#3A86FF"
    ),
    labels = c(
      "HDR"            = "HDR",
      "µ ± σ"       = expression(mu ~ "±" ~ sigma)
    )
  ) +
  labs(x = "x", y = "Density", fill = NULL) +
  theme_minimal(base_size = 15) +
  theme(
    strip.text        = element_text(size = 15, face = "bold"),
    axis.title        = element_text(size = 17),
    axis.text         = element_text(size = 12),
    legend.position   = c(0.93, 0.75),
    legend.text       = element_text(size = 13),
    legend.background = element_rect(fill = "white", colour = "grey80")
  )
p_hdr


ggsave('splash.pdf', width = 16, height = 4, dpi = 300)

############################# Cauchy figure ####################################

x <- seq(-4, 4, length.out = 1000)
scales <- c(0.5, 1, 2)

df <- expand.grid(x = x, gamma = scales)
df$y <- dcauchy(df$x, location = 0, scale = df$gamma)

p_cauchy <- ggplot(df, aes(x = x, y = y, colour = factor(gamma))) +
  geom_line(linewidth = 1) +
  scale_colour_d3(name = "Scale", 
                  labels = parse(text = paste0("gamma == ", scales))) +
  labs(x = "x", y = "Density") +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 17),
    axis.text  = element_text(size = 13),
    legend.title = element_text(size = 16),
    legend.text  = element_text(size = 13),
    legend.position = c(0.84, 0.74),
    legend.background = element_rect(fill = "white", colour = "grey80")
  )


############################### Levelling figure ###############################

x_grid <- seq(-5, 5, length.out = 3000)
dx <- diff(x_grid)[1]

# Original density: bimodal, with a high left peak and lower right peak
p <- 
  0.72 * dnorm(x_grid, mean = -2.0, sd = 0.45) +
  0.28 * dnorm(x_grid, mean =  1.7, sd = 0.65)

# Regions A and B
A <- x_grid > -2.35 & x_grid < -1.65
B <- x_grid > -1.0  & x_grid <  0.8

# Flatten A to inf_A p, then put all displaced mass uniformly into B
inf_A_p <- min(p[A])
removed_mass <- sum((p[A] - inf_A_p) * dx)

p_prime <- p
p_prime[A] <- inf_A_p
p_prime[B] <- p_prime[B] + removed_mass / sum(B * dx)

# Data for plotting
df <- tibble(
  x = rep(x_grid, 2),
  density = c(p, p_prime),
  distribution = factor(
    rep(c("original", "levelled"), each = length(x_grid)),
    levels = c("original", "levelled")
  )
)

regions <- tibble(
  xmin = c(-2.35, -1.0),
  xmax = c(-1.65,  0.8),
  region = c("A", "B")
)

line_df <- tibble(
  distribution = factor(
    c("original", "original", "levelled", "levelled"),
    levels = c("original", "levelled")
  ),
  y = c(
    min(p[A]),
    max(p[B]),
    min(p_prime[A]),
    max(p_prime[B])
  ),
  region = c("A", "B", "A", "B"),
  label = c(
    "inf[A]~italic(p)",
    "sup[B]~italic(p)",
    "inf[A]~italic(p)*\"'\"",
    "sup[B]~italic(p)*\"'\""
  ),
  x = rep(-4.5, 4),
  vjust = c(-0.45, -0.45, -0.45, -0.45)
)

p_level <- ggplot(df, aes(x = x, y = density)) +
  geom_rect(
    data = regions,
    aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = region),
    inherit.aes = FALSE,
    alpha = 0.18
  ) +
  geom_line(linewidth = 1.1) +
  geom_hline(
    data = line_df,
    aes(yintercept = y, colour = region),
    linetype = "dashed",
    linewidth = 0.8
  ) +
  geom_text(
    data = line_df,
    aes(x = x, y = y, label = label, colour = region, vjust = vjust),
    parse = TRUE,
    hjust = 0,
    size = 4
  ) +
  facet_wrap(
    ~ distribution,
    ncol = 2,
    labeller = labeller(
      distribution = as_labeller(c(
        original = "Original~density~italic(p)",
        levelled = "Levelled~density~italic(p)*\"'\""
      ), default = label_parsed)
    )
  ) +
  scale_fill_manual(
    values = c(A = "tomato", B = "skyblue"),
    labels = c(A = "A: mass removed", B = "B: mass added")
  ) +
  scale_colour_manual(
    values = c(A = "tomato", B = "skyblue"),
    guide = "none"
  ) +
  labs(
    x = "x",
    y = "Density",
    fill = NULL
  ) +
  coord_cartesian(ylim = c(0, 0.7)) +
  theme_minimal(base_size = 15) +
  theme(
    legend.position = c(0.98, 0.98),
    legend.justification = c(1, 1),
    legend.background = element_rect(fill = "white", colour = "grey80"),
    strip.text = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 17),
    axis.text = element_text(size = 13),
    legend.text = element_text(size = 13)
  )
p_level

### Combine the figures ###
combined_plot <- plot_grid(
  p_cauchy, p_level,
  labels = c("(A)", "(B)"),
  nrow = 1,
  align = "hv",
  axis = "tb",
  rel_widths = c(1, 2)
)

combined_plot

ggsave(
  "levelling.pdf",
  combined_plot,
  width = 16,
  height = 4,
  dpi = 300
)



