# Yavuz, Y. E., & Kahraman, F. (2024). Evaluation of the prediagnosis and management of ChatGPT-4.0 in clinical cases in cardiology. Future cardiology, 20(4), 197–207. https://doi.org/10.1080/14796678.2024.2348898
# Modificado
CC_prueba_1_adaptado = """
A 49-year-old male patient presents to the emergency department at around 5:00 AM
with oppressive chest pain that woke him from sleep. He reports sweating and mild
shortness of breath. He has a known history of hypertension and active smoking.

Vital signs: blood pressure 138/86 mmHg, heart rate 96 bpm, respiratory rate 20/min,
oxygen saturation 96%, temperature 36.8°C.

ECG findings: ST depression in leads V1-V2, dominant R wave in V2, and 1.5 mm ST
elevation in posterior leads V7-V8-V9.

Relevant cardiovascular risk variables:
age=49, sex=male, chest pain type=typical angina, resting blood pressure=138,
cholesterol=245 mg/dL, fasting blood sugar=0, resting ECG=ST-T abnormality,
maximum heart rate=150 bpm, exercise-induced angina=1, oldpeak=2.1,
slope=flat, ca=0, thal=reversible defect.

The patient is not currently taking beta blockers or anticoagulants.
"""


CC_prueba_2_adaptado = """
A 57-year-old male patient with chronic kidney disease on routine hemodialysis
through a long-standing arterio-venous fistula presents to the emergency department
with progressive shortness of breath, chest pain, weakness, sweating and poor general
condition over the last 2 weeks. He reports worsening dyspnea over the last 15 days
and unintentional weight loss during the last month.

Past medical history: chronic kidney disease, diabetes mellitus type 2, hypertension,
long-term dialysis through arterio-venous fistula.

Vital signs: blood pressure 105/70 mmHg, heart rate 115 bpm, respiratory rate 28/min,
temperature 39.0°C, oxygen saturation 91%.

Physical examination: the patient appears weak, tachypneic and in poor general
condition. Lung examination reveals rales at both bases. Cardiac examination reveals
regular tachycardia, normal S1 and S2, and a grade III/VI holosystolic murmur best
heard at the lower left sternal border. There is mild peripheral edema.

Relevant cardiovascular risk variables:
age=57, sex=male, chest pain type=atypical angina, resting blood pressure=105,
cholesterol=210 mg/dL, fasting blood sugar=1, resting ECG=non-specific ST-T changes,
maximum heart rate=115 bpm, exercise-induced angina=0, oldpeak=1.0,
slope=flat, ca=1, thal=fixed defect.

Current medication: insulin, amlodipine and furosemide.
"""


CC_prueba_3_adaptado = """
An 82-year-old female patient presents with progressive shortness of breath,
orthopnea and peripheral edema one month after single-chamber pacemaker implantation.
She had previously presented with syncope and symptomatic sinus pauses on Holter ECG.

Past medical history: hypertension, type 2 diabetes mellitus, paroxysmal atrial
fibrillation and moderate functional mitral regurgitation. Echocardiography before
pacemaker implantation showed preserved left ventricular ejection fraction, mild
tricuspid regurgitation and mildly dilated left atrium.

Before the procedure, ECG showed atrial fibrillation with a ventricular rate of
121 bpm. A single-chamber pacemaker was implanted through a pathway to the right
ventricular apex. Beta blockers were restarted after discharge.

Current vital signs: blood pressure 150/88 mmHg, heart rate 104 bpm, respiratory
rate 24/min, oxygen saturation 93%, temperature 36.7°C.

Current physical examination: elevated jugular venous pressure, bilateral basal rales
and peripheral edema, consistent with heart failure syndrome.

Relevant cardiovascular risk variables:
age=82, sex=female, chest pain type=asymptomatic, resting blood pressure=150,
cholesterol=230 mg/dL, fasting blood sugar=1, resting ECG=atrial fibrillation,
maximum heart rate=121 bpm, exercise-induced angina=0, oldpeak=0.8,
slope=flat, ca=1, thal=normal.

Current medication: beta blocker, oral anticoagulant and antihypertensive treatment.
"""

casos_de_prueba=[CC_prueba_1_adaptado, CC_prueba_2_adaptado, CC_prueba_3_adaptado]