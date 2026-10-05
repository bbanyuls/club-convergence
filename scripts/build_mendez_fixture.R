# Build tests/fixtures/mendez2020_stata.csv from Mendez (2020), "Convergence
# Clubs in Labor Productivity and its Proximate Sources" (Springer), whose data
# and code are published under the MIT licence at
#   https://github.com/quarcs-lab/mendez2020-convergence-clubs-code-data
#
# The R data files of that repository contain, for 26 high-income (hi1990 =
# yes) and 82 other countries, 1990-2014, the HP-filtered logs of labor
# productivity, capital-output ratio, human capital and TFP (PWT 9.0) and the
# club memberships found with Stata's psecta (club_*, kq(0.333)) and after
# imergeclub (finalclub_*). The Python test-suite checks that
# find_clubs(convention="Stata", time_trim=0.333) reproduces them.
#
# Usage (from the repository root):
#   git clone https://github.com/quarcs-lab/mendez2020-convergence-clubs-code-data /tmp/mendez
#   (cd /tmp/mendez/assets && unzip -o dat_hiYes.zip && unzip -o dat_hiNo.zip)
#   Rscript scripts/build_mendez_fixture.R /tmp/mendez/assets

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[1] else "/tmp/mendez/assets"
vars <- c("log_lp", "log_ky", "log_h", "log_tfp")
keep <- c("country", "year", vars, paste0("club_", vars), paste0("finalclub_", vars))

out <- do.call(rbind, lapply(c(yes = "hiYes", no = "hiNo"), function(s) {
    e <- new.env()
    load(file.path(root, sprintf("dat_%s/ExPanD_nb_data.Rdata", s)), e)
    d <- e$nb_df[, keep]
    d$country <- as.character(d$country)
    cbind(sample = s, d)
}))
out <- out[order(out$sample, out$country, out$year), ]
write.csv(out, "tests/fixtures/mendez2020_stata.csv", row.names = FALSE, na = "")
cat("wrote", nrow(out), "rows\n")
