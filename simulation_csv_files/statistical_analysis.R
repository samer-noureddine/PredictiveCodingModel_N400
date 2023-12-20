library(lme4)
library(lmerTest)
library(tidyverse)
library(emmeans)
library(effectsize)


# prevent R from wrapping its output around too soon
options(width=300)

nrm <- function(x) {
  return((x - mean(x))/sd(x))
}


base = "D:\\PROJECTS\\PC_Paper\\PCLex_N400_Cognition\\simulation_csv_files\\"
plotloc = "D:\\PROJECTS\\PC_Paper\\PCLex_N400_Cognition\\plots\\stats_plots\\"

# load data
std_sim= read_csv(paste(base, "Standard_Simulation_N400_2_to_11.csv", sep = ""))
wrd_psd_sim= read_csv(paste(base, "Word_vs_Pseudoword_Simulation_N400_2_to_11.csv", sep = ""))
reppriming_sim= read_csv(paste(base, "RepetitionPriming_Simulation_N400_24_to_33.csv", sep = ""))
sempriming_sim = read_csv(paste(base, "SemanticPriming_Simulation_N400_24_to_33.csv", sep = ""))
cloze_sim = read_csv(paste(base, "ClozeProbability_Simulation_N400_22_to_31.csv", sep = ""))
lexviol_sim = read_csv(paste(base, "LexicalViolation_Simulation_N400_22_to_31.csv", sep = ""))
sempredoverlap_sim= read_csv(paste(base, "SemanticPredictionOverlap_Simulation_N400_22_to_31.csv", sep = ""))
orthpredoverlap_sim= read_csv(paste(base, "OrthographicPredictionOverlap_Simulation_N400_22_to_31.csv", sep = ""))

# normalize all quantitative variables
nrm_std_sim = std_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_reppriming_sim = reppriming_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_sempriming_sim = sempriming_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_cloze_sim = cloze_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_lexviol_sim = lexviol_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_sempredoverlap_sim = sempredoverlap_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_orthpredoverlap_sim = orthpredoverlap_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_wrd_psd_sim = wrd_psd_sim %>% mutate_if(is.numeric, list(norm = nrm))
options(width = 300)

######## stats ########

# Section 1.1, 1.3, 1.4
std_model = lm(LexSemErr ~ ONsize_norm + Frequency_norm + Concreteness, data = nrm_std_sim)
summary(std_model)

# Section 1.2: in the paper, the outcome variable is LexSemErr_norm.
psd_ONsize_model_norm = lm(LexSemErr_norm ~ ONsize_norm*IsWord_code, data = nrm_wrd_psd_sim) # this is in the paper
summary(psd_ONsize_model_norm)

# instead, I should use `LexSemErr` itself, like this:
psd_ONsize_model = lm(LexSemErr ~ ONsize_norm*IsWord_code, data = nrm_wrd_psd_sim) 
summary(psd_ONsize_model)

# Section 2.1
RepPrimModel_N400 = lmer(LexSemErr ~ Repeated_code + ( 0 + Repeated_code | Word), data = nrm_reppriming_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
# failed to converge:
# RepPrimModel = lmer(LexSemErr ~ Repeated_code + (Repeated_code | Word), data = nrm_reppriming_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
# RepPrimModel = lmer(LexSemErr ~ Repeated_code + (Repeated_code || Word), data = nrm_reppriming_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(RepPrimModel_N400)

options(width = 100)
nrm_sempriming_sim %>% glimpse()

# Section 2.2
SemPrimModel_N400= lmer(LexSemErr ~ SemanticRelatedness_code + (0 + SemanticRelatedness_code | Word), data = nrm_sempriming_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(SemPrimModel_N400)

# # difference between repeated and semantically primed words
repprimed = nrm_reppriming_sim %>% filter(Repeated_name == "repeated")%>% mutate(dummy_condition = "repeated")
semprimed = nrm_sempriming_sim %>% filter(SemanticRelatedness_name == "semrelated") %>% mutate(dummy_condition = "semrelated")
sem_vs_rep = full_join(semprimed, repprimed, by = "WordInds")
t.test(sem_vs_rep$LexSemErr.x, sem_vs_rep$LexSemErr.y, paired = TRUE)

# Section 3.1
ClzModel_N400 = lmer(LexSemErr ~ Cloze_code_norm + (0 + Cloze_code_norm  | Word), data = nrm_cloze_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(ClzModel_N400)

# Section 3.2
nrm_lexviol_sim_unexp = nrm_lexviol_sim %>% filter(IsExpected_code == -1)
LexViolModel_N400 = lmer(LexSemErr ~ Constraint_code  + (Constraint_code || Word), data = nrm_lexviol_sim_unexp, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(LexViolModel_N400)

# Section 3.3
PE_expected = nrm_cloze_sim %>% mutate(dummy_condition = "expected") %>% filter(Cloze_name == "high_cloze")
unexp_related_sempredoverlap = nrm_sempredoverlap_sim %>% filter(Cloze_name == "HighCloze")%>% filter(Relatedness_name == "SemRelated") %>% mutate(dummy_condition = "unexp_related")
unexp_unrelated_sempredoverlap = nrm_sempredoverlap_sim %>% filter(Cloze_name == "HighCloze")%>% filter(Relatedness_name == "SemUnrelated") %>% mutate(dummy_condition = "unexp_unrelated")
temp = full_join(PE_expected, unexp_related_sempredoverlap)
final_SPO = full_join(temp, unexp_unrelated_sempredoverlap)
# create the three conditions: expected, overlap, nonoverlap
final_SPO = final_SPO %>% mutate(dummy_condition = fct_relevel(dummy_condition,"unexp_related","unexp_unrelated", "expected"))
expected_semoverlap_nonoverlap_model_N400 = lmer(LexSemErr ~ as.factor(dummy_condition) + (1| Word), data = final_SPO, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000))) # nolint: line_length_linter.
summary(expected_semoverlap_nonoverlap_model_N400)

# Section 3.4: Effect of contextual constraint on the anticipatory semantic overlap effect
unexp_related_sempredoverlap_99 = nrm_sempredoverlap_sim %>% filter(Cloze_name == "HighCloze")%>% filter(Relatedness_name == "SemRelated") %>% mutate(dummy_condition = "unexp_related")
unexp_unrelated_sempredoverlap_99 = nrm_sempredoverlap_sim %>% filter(Cloze_name == "HighCloze")%>% filter(Relatedness_name == "SemUnrelated") %>% mutate(dummy_condition = "unexp_unrelated")
unexp_related_sempredoverlap_50 = nrm_sempredoverlap_sim %>% filter(Cloze_name == "MedCloze")%>% filter(Relatedness_name == "SemRelated") %>% mutate(dummy_condition = "unexp_related")
unexp_unrelated_sempredoverlap_50 = nrm_sempredoverlap_sim %>% filter(Cloze_name == "MedCloze")%>% filter(Relatedness_name == "SemUnrelated") %>% mutate(dummy_condition = "unexp_unrelated")
temp2 = full_join(unexp_related_sempredoverlap_99, unexp_related_sempredoverlap_50)
temp3 = full_join(unexp_unrelated_sempredoverlap_99, unexp_unrelated_sempredoverlap_50)
finalSPO_constraint = full_join(temp2, temp3)
finalSPO_constraint = finalSPO_constraint %>% mutate(Relatedness_name = fct_relevel(Relatedness_name, "SemUnrelated", "SemRelated"))%>% mutate(Cloze_name = fct_relevel(Cloze_name, "MedCloze", "HighCloze"))
SPOconstraint_model_N400 = lmer(LexSemErr ~ as.factor(Cloze_name)*as.factor(Relatedness_name) + (1| Word), data = finalSPO_constraint, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(SPOconstraint_model_N400)

finalSPO_constraint %>% group_by(Relatedness_name, Cloze_name) %>% summarize(n())

options(width=300)


# Section 3.5
unexp_related_orthpredoverlap= nrm_orthpredoverlap_sim %>% filter(IsNeighborofExpected_name == "Neighbor" & IsWord_name == "Word")%>% mutate(dummy_condition = "unexp_related")
unexp_unrelated_orthpredoverlap= nrm_orthpredoverlap_sim %>% filter(IsNeighborofExpected_name == "NonNeighbor" & IsWord_name == "Word") %>% mutate(dummy_condition = "unexp_unrelated")
PE_expected = nrm_cloze_sim  %>% filter(Cloze_name == "high_cloze") %>% mutate(dummy_condition = "expected")
unexpecteds = full_join(unexp_related_orthpredoverlap, unexp_unrelated_orthpredoverlap)
finalOPO = full_join(PE_expected, unexpecteds)
finalOPO = finalOPO %>% mutate(dummy_condition = fct_relevel(dummy_condition,"unexp_related","unexp_unrelated", "expected"))
OPO_model= lmer(LexSemErr ~ as.factor(dummy_condition) + (1 | Word) , data = finalOPO, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(OPO_model)

# Section 3.6
OrthPredOverlapModel = lmer(LexSemErr ~ IsNeighborofExpected_code*IsWord_code + (0 + IsNeighborofExpected_code | Word), data = nrm_orthpredoverlap_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(OrthPredOverlapModel)

# Section 4.1
Interactions_RepPrimModel = lmer(LexSemErr ~ ONsize_norm*Repeated_code + Frequency_norm*Repeated_code + Concreteness*Repeated_code + ( 0 + Repeated_code | Word), data = nrm_reppriming_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(Interactions_RepPrimModel)

# Section 4.2
split_nrm_cloze_sim = nrm_cloze_sim %>% filter(Cloze_name == 'high_cloze' | Cloze_name == 'low_cloze')
Interactions_ClzModel = lmer(LexSemErr ~ ONsize_norm*Cloze_code_norm  + Frequency_norm*Cloze_code_norm  + Concreteness*Cloze_code_norm  + (Cloze_code_norm  || Word), data = split_nrm_cloze_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(Interactions_ClzModel)
# failed to converge:
# splitClzModel = lmer(LexSemErr ~ ONsize_norm*Cloze_code  + Frequency_norm*Cloze_code  + Concreteness*Cloze_code  + (Cloze_code  | Word), data = split_nrm_cloze_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
