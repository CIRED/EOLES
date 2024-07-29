library(haven)
library(tidyverse)
library(maps)
library(ggplot2)
library(EnvStats)
library(lmtest)
library(dplyr)
library(DescTools)
library(skimr)
library(DataExplorer)
library(gt)



df = data.frame(grand_transport = c(21, 15, 19, 14, 11, 9),
                reseaux_regionnaux = c(28, 29, 24, 19, 13, 10),
                adaptation_reseaux = c(105, 117, 92, 84, 73, 67),
                total_invest_rsx = c(154, 161, 135, 117, 97, 86),
                terrestre = c(76, 71, 84, 66, 53, 86.7),
                offshore = c(62, 56.5, 76, 53, 37, 90.9),
                eolien = c(138, 127.5, 160, 119, 90, 177.6),
                solaire = c(220, 262, 150, 139, 93, 85.7),
                nuc_hist = c(0, 1.6, 1.6, 1.6, 1.6, 4.6),
                epr2 = c(0, 0, 0, 19.8, 36.4, 39.6),
                smr = c(0, 0, 0, 0, 0, 6.3),
                h2 = c(32.8, 33.7, 30.2, 16.6, 4.6, 0),
                batteries = c(36.5, 39.9, 20.9, 12.0, 2.1, 0.5))


reg = lm(total_invest_rsx ~ eolien + solaire + h2, df)
summary(reg)
