# Generate reference results with the R package ConvergenceClubs.
#
# The Python test-suite compares its output against these fixtures.
# Usage (from the repository root):
#     Rscript scripts/generate_r_fixtures.R
# Requires: install.packages(c("ConvergenceClubs", "mFilter", "jsonlite"))

suppressPackageStartupMessages({
    library(ConvergenceClubs)
    library(jsonlite)
})

out_file <- "tests/fixtures/r_reference.json"

model_list <- function(m) {
    list(beta = unname(m["beta"]), std_err = unname(m["std.err"]),
         t_value = unname(m["tvalue"]), p_value = unname(m["pvalue"]))
}

clubs_list <- function(cl) {
    nm <- setdiff(names(cl), "divergent")
    list(
        clubs = lapply(nm, function(n) {
            x <- cl[[n]]
            list(name = n,
                 ids = as.integer(x$id) - 1L,
                 model = model_list(x$model),
                 cstar = if (is.null(x$cstar)) NULL else x$cstar,
                 merged_from = if (is.null(x$clubs)) NULL else I(x$clubs))
        }),
        divergent = I(if (is.null(cl$divergent)) integer(0) else as.integer(cl$divergent$id) - 1L)
    )
}

# Simulated panel with heterogeneous transition paths, to exercise branches
# (divergent units, incremental c*) that the GDP data does not reach.
simulate_panel <- function(seed, n = 40, t = 50) {
    set.seed(seed)
    tt <- seq_len(t)
    groups <- sample(1:4, n, replace = TRUE)
    limits <- c(1.3, 1.0, 0.8, 0.6)
    x <- t(sapply(seq_len(n), function(i) {
        start <- runif(1, 0.5, 1.5)
        speed <- runif(1, 0.05, 0.3)
        path <- limits[groups[i]] + (start - limits[groups[i]]) * exp(-speed * tt)
        drift <- if (i %% 13 == 0) 0.01 * tt else 0
        10 + path + drift + cumsum(rnorm(t, sd = 0.002))
    }))
    df <- data.frame(unit = sprintf("u%02d", seq_len(n)), x, stringsAsFactors = FALSE)
    colnames(df) <- c("unit", paste0("t", tt))
    df
}

data("filteredGDP")
data("GDP")

datasets <- list(
    filtered_gdp = filteredGDP,
    sim1 = simulate_panel(1),
    sim2 = simulate_panel(2, n = 30, t = 30),
    sim3 = simulate_panel(3, n = 60, t = 40)
)

fixtures <- list(datasets = list(), log_t = list(), find_clubs = list(), merge_clubs = list())

for (dn in names(datasets)) {
    df <- datasets[[dn]]
    if (dn != "filtered_gdp") {
        fixtures$datasets[[dn]] <- list(units = df[, 1], values = unname(as.matrix(df[, -1])))
    }
}

# hp filter of log(GDP), lambda = 400 (how filteredGDP was built)
hp <- t(apply(GDP[, -1], 1, function(x) mFilter::hpfilter(log(x), freq = 400, type = "lambda")$trend))
fixtures$hp_filter <- list(lambda = 400, trend = unname(hp))

# log t regression on several subsets of the GDP data
X <- filteredGDP[, -1]
subsets <- list(all = seq_len(nrow(X)), first2 = 1:2, first10 = 1:10,
                every3 = seq(1, nrow(X), by = 3), last40 = (nrow(X) - 39):nrow(X))
for (sn in names(subsets)) {
    for (hac in c("FQSB", "AQSB")) {
        for (tt in c(1/3, 0.2, 0.5)) {
            H <- computeH(X[subsets[[sn]], ])
            fixtures$log_t[[length(fixtures$log_t) + 1]] <- list(
                subset = sn, ids = as.integer(subsets[[sn]]) - 1L, hac_method = hac,
                time_trim = tt, model = model_list(estimateMod(H, tt, HACmethod = hac)))
        }
    }
}

configs <- list(
    list(hac = "FQSB", trim = 1/3, cstar = 0, cm = "fixed"),
    list(hac = "AQSB", trim = 1/3, cstar = 0, cm = "fixed"),
    list(hac = "FQSB", trim = 0.3, cstar = 0, cm = "fixed"),
    list(hac = "FQSB", trim = 1/3, cstar = -1, cm = "fixed"),
    list(hac = "FQSB", trim = 1/3, cstar = 0, cm = "incremental"),
    list(hac = "AQSB", trim = 1/3, cstar = 0, cm = "incremental")
)

for (dn in names(datasets)) {
    df <- datasets[[dn]]
    dc <- 2:ncol(df)
    for (cfg in configs) {
        cl <- findClubs(df, dataCols = dc, unit_names = 1, refCol = ncol(df),
                        time_trim = cfg$trim, HACmethod = cfg$hac, cstar = cfg$cstar,
                        cstar_method = cfg$cm, cstar_increment = 0.1, cstar_cap = 3)
        args <- list(dataset = dn, hac_method = cfg$hac, time_trim = cfg$trim,
                     cstar = cfg$cstar, cstar_method = cfg$cm)
        fixtures$find_clubs[[length(fixtures$find_clubs) + 1]] <- c(list(args = args), clubs_list(cl))

        for (mm in c("PS", "vLT")) {
            for (md in c(FALSE, TRUE)) {
                mcl <- suppressMessages(mergeClubs(cl, mergeMethod = mm, mergeDivergent = md))
                margs <- c(args, list(method = mm, merge_divergent = md))
                fixtures$merge_clubs[[length(fixtures$merge_clubs) + 1]] <-
                    c(list(args = margs), clubs_list(mcl))
            }
        }
    }
}

dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
write_json(fixtures, out_file, auto_unbox = TRUE, digits = I(17), null = "null")
cat("wrote", out_file, "\n")
