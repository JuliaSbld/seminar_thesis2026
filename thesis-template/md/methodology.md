# Methodology

## Research Model

Arising from the hypotheses stated in the prior section, the following model [(see Figure \ref{fig:1}) guided my research going forward.

Research Model \label{fig:1}{width=80%}

The overlaying effect of AI disclosure on message credibility (H1a,b) is the baseline effect, which is then split up by the mediator “perceived AI Usage” going forward.
The three AI disclosure levels serve as independent variables here, which affect the perceived AI usage (H2a,b,c).
Perceived AI usage is then hypothesized to influence the message credibility (H3).
Critical Media Literacy is added as a moderator of the relationship of disclosure and perceived AI usage (H4a,b,c).
AI Attitude is hypothesized to moderate the connection between perceived AI usage and message credibility (H5).

## Research Design

The research conducted within this seminar paper is a quantitative experiment using a within-subject design to explore the research question of how different types of AI disclosure influence the message credibility of Social Media Posts.
Seven dummy Social Media Posts have been designed as stimuli and marked with three different types of AI disclosure.
Further, the Posts did not have a clear source or author in order to isolate the message credibility from other factors like source credibility and author credibility [@appelman_measuring_2016].
The posts were ‘created with AI’, ‘partly created with AI’, or not marked at all.
Three posts did not have a label, and two were labeled with the two different AI labels, respectively. This assignment of a label to a post is static across participants.
The disclosure is used as the independent variable within the experiment, and all participants are exposed to all three levels of disclosure.
Message Credibility is used as the dependent variable in the experiment and is being measured using the scale proposed by @appelman_measuring_2016, consisting of three adjectives (accurate, authentic, believable) that are being rated on a seven-point Likert scale.
To explore other possible personal factors influencing the message credibility of the stimuli, the critical media consumption, personal AI Attitude, as well as some demographic information, are being measured.

## Instrument

The Experiment is designed in four overall blocks.
Firstly, the participants are exposed to the seven dummy Posts one by one. After each, the message credibility as well as their perception of AI usage in the Post are questioned.
The second block uses the Critical Media Literacy scale by @koc_development_2016 to evaluate how participants perceive their consumption of media.
The scale consists of eleven statements, which have been classified on a seven-point Likert scale from completely disagree to completely agree.
An attention checker has been added to this block of the questionnaire to ensure attentive answers from the participants.
After that, participants are presented with the AIAS-4 to question their attitude towards AI [@grassini_development_2023]. This scale uses four statements on a seven-point Likert scale as well.
At last, demographics including age, gender, highest level of education, and professional field are documented.

## Population, Sample, and Data Collection

Data for this research has been collected in the time from the 07.05.2026 until 08.06.2026.
Completing the questionnaire took about five to ten minutes, and participants completed it without supervision on their own devices.
Half of the participants were sourced through personal connections, and the other half were sourced through Prolific.
All participants agreed to their data being used within the context of this research anonymously.
In the context of the seminar, a preregistration of the research was not included.
The questionnaire has been completed by 20 participants, of whom 10 were male and 10 were female.
The ages of participants range from 19 to 59, with a median age of 32.6.
Only data from completed experiments have been used within the analysis of this research.
In this context, one of the original 20 participants was excluded due to missing values in the critical media consumption block.

The number of 19 analyzed participants limits the research within this paper to be rather exploratory and in need of further confirmation in future research.
Nonetheless, several strands of interesting factors influencing message credibility in AI-mediated content are being explored with this small sample.

## Data analysis

In order to account for the within-subject variation of stimuli, the data are analyzed using a mixed-effect model [@baayen_mixed-effects_2008] whenever possible.
This model accounts for random intercepts of participants, meaning a differing baseline of credibility perception across participants is taken into consideration.
Two exceptions to this had to be made for two different reasons.
The analysis of Critical Media Literacy as a moderator for the relationship of AI disclosure and perceived AI Usage was changed to an OLS regression because the between-person variance was too small for the mixed-effects model, resulting in convergence failure.
Further, the mediation analysis could not be carried out with the mixed-effect model and has also been changed to an OLS regression.
This change has been made for technical reasons, as the statsmodels mediation does not support mixed-effect models.
The limitation is discussed further in a later chapter.

## Reliability and validity in this study

The reliability of the scales used in the context of this research has been computed using Cronbach’s Alpha.
For the Message Credibility using the three-item scale, a score of $\alpha$  = .944 can be reported, even higher than the $\alpha$  = .87 reported in the original construction of the scale [@appelman_measuring_2016].
The AI Attitude scale shows an excellent value of $\alpha$  = .900 as well.
At last, the critical media consumption scale consisting of 11 items has a Cronbach’s $\alpha$ = 0.827, close to the $\alpha$  = .87 reported by @koc_development_2016.
Therefore, the reliability of all three used scales holds within this research.
