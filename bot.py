import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler, ConversationHandler
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
from pycaret.classification import setup, compare_models, predict_model
import joblib
import os

======================

CONFIG

======================

BOT_TOKEN = os.getenv("BOT_TOKEN")  # on récupère la variable d'environnement

if not BOT_TOKEN:
raise ValueError("❌ Erreur : la variable d'environnement BOT_TOKEN n'est pas définie !")

logging.basicConfig(level=logging.INFO)

======================

Historique

======================

DATA_FILE = "historique.csv"
if os.path.exists(DATA_FILE):
historique = pd.read_csv(DATA_FILE)
else:
colonnes_scores = [f"score{i}" for i in range(1,16)] + [f"cote_score{i}" for i in range(1,16)]
colonnes_double_chance = ["double1X","cote1X","double12","cote12","doubleX2","coteX2"]
colonnes_extra = ["diff_coteA_B","ratio_double1X_X2"]
historique = pd.DataFrame(columns=["equipeA","equipeB","coteA","coteN","coteB"] +
colonnes_double_chance + colonnes_scores + colonnes_extra +
["prediction_p1","prediction_finale","resultat","score_reel"])

======================

Modèles IA

======================

models = {
"RandomForest": RandomForestClassifier(n_estimators=50),
"LogisticRegression": LogisticRegression(max_iter=200),
"NaiveBayes": GaussianNB(),
"KNN": KNeighborsClassifier(n_neighbors=3),
"XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
"LightGBM": lgb.LGBMClassifier()
}

scalers = {
"LogisticRegression": StandardScaler(),
"KNN": StandardScaler(),
"XGBoost": StandardScaler(),
"LightGBM": StandardScaler()
}

model_weights = {name:1.0 for name in models.keys()}

======================

Fonctions IA

======================

def generer_features(cotes, cotes_scores, cotes_double):
diff_cote = cotes[0] - cotes[2]  # coteA - coteB
ratio_double = cotes_double[0] / (cotes_double[2]+0.01)
return cotes + cotes_double + cotes_scores + [diff_cote, ratio_double]

def entrainer_models():
global model_weights
if len(historique) < 5:
logging.info("Pas assez de données pour entraîner les modèles.")
return
feature_cols = ["coteA","coteN","coteB","double1X","double12","doubleX2"] + \
[f"cote_score{i}" for i in range(1,16)] + ["diff_coteA_B","ratio_double1X_X2"]
X = historique[feature_cols]
y = historique["resultat"]

for name, model in models.items():
try:
X_train = X.copy()
if name in scalers:
X_train = scalers[name].fit_transform(X_train)
model.fit(X_train, y)
logging.info(f"{name} entraîné avec succès.")
except Exception as e:
logging.warning(f"Erreur entraînement {name}: {e}")
continue

Calcul poids dynamiques

for name, model in models.items():
try:
X_eval = X.copy()
if name in scalers:
X_eval = scalers[name].transform(X_eval)
acc = model.score(X_eval, y)
model_weights[name] = max(0.1, acc)
except Exception as e:
logging.warning(f"Erreur calcul poids {name}: {e}")
model_weights[name] = 0.1

def prediction_ensemble(features):
if len(historique)<5:
return "gagnant", 60
X_input = pd.DataFrame([features])
votes = {"gagnant":0,"perdant":0}
total_weight = 0
for name, model in models.items():
try:
X_in = X_input.copy()
if name in scalers:
X_in = scalers[name].transform(X_in)
pred = model.predict(X_in)[0]
votes[pred] += model_weights[name]
total_weight += model_weights[name]
except Exception as e:
logging.warning(f"Erreur prediction {name}: {e}")
continue
classe = max(votes,key=votes.get)
pourcentage = (votes[classe]/total_weight)*100 if total_weight>0 else 60
return classe, round(pourcentage,1)

def choix_score_pondere(scores, cotes_scores, classe):
probs = np.array([1/c for c in cotes_scores])
probs /= probs.sum()
if classe=="gagnant":
probs[:3] *= 1.3
probs /= probs.sum()
return np.random.choice(scores, p=probs)

def scores_plus_probables(scores, cotes_scores, top_n=2):
probs = np.array([1/c for c in cotes_scores])
probs /= probs.sum()
indices = np.argsort(probs)[::-1][:top_n]
result = [(scores[i], round(probs[i]*100,1)) for i in indices]
return result

# ======================
# États conversation
# ======================
ENTREE_EQUIPE, ENTREE_COTES, ENTREE_DOUBLE, ENTREE_SCORES, ENTREE_COTES_SCORES, RESULTAT, SCORE_REEL = range(7)

# ======================
# Handlers Telegram
# ======================

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Bienvenue ! Envoie-moi le nom des équipes sous la forme : EquipeA,EquipeB"
    )
    return ENTREE_EQUIPE

async def recevoir_equipes(update: Update, context: CallbackContext):
    try:
        equipes = [x.strip() for x in update.message.text.split(",")]
        if len(equipes) != 2:
            raise ValueError
        context.user_data["equipeA"] = equipes[0]
        context.user_data["equipeB"] = equipes[1]
    except:
        await update.message.reply_text("⚠️ Format invalide. Exemple : PSG,Real Madrid")
        return ENTREE_EQUIPE
    await update.message.reply_text(
        "Envoie maintenant les cotes 1X2 sous la forme : coteA,coteN,coteB"
    )
    return ENTREE_COTES

async def recevoir_cotes(update: Update, context: CallbackContext):
    try:
        cotes = [float(x) for x in update.message.text.split(",")]
        if len(cotes) != 3:
            raise ValueError
        context.user_data["cotes"] = cotes
    except:
        await update.message.reply_text("⚠️ Format invalide. Exemple : 2.1,3.5,2.8")
        return ENTREE_COTES
    await update.message.reply_text(
        "Envoie maintenant les cotes double chance dans l'ordre : 1X,12,X2 (séparées par des virgules)"
    )
    return ENTREE_DOUBLE

async def recevoir_double(update: Update, context: CallbackContext):
    try:
        cotes_double = [float(x) for x in update.message.text.split(",")]
        if len(cotes_double) != 3:
            raise ValueError
        context.user_data["cotes_double"] = cotes_double
    except:
        await update.message.reply_text("⚠️ Format invalide. Exemple : 1.3,1.5,1.4")
        return ENTREE_DOUBLE
    await update.message.reply_text(
        "Envoie maintenant les 15 scores exacts proposés (séparés par des virgules)"
    )
    return ENTREE_SCORES

async def recevoir_scores(update: Update, context: CallbackContext):
    scores = [s.strip() for s in update.message.text.split(",")]
    if len(scores) != 15:
        await update.message.reply_text("⚠️ Tu dois envoyer exactement 15 scores.")
        return ENTREE_SCORES
    context.user_data["scores"] = scores
    await update.message.reply_text(
        "Envoie maintenant les 15 cotes correspondantes (séparées par des virgules)"
    )
    return ENTREE_COTES_SCORES

async def recevoir_cotes_scores(update: Update, context: CallbackContext):
    try:
        cotes_scores = [float(x) for x in update.message.text.split(",")]
        if len(cotes_scores) != 15:
            raise ValueError
        context.user_data["cotes_scores"] = cotes_scores
    except:
        await update.message.reply_text("⚠️ Format invalide. Exemple : 6.0,5.5,...")
        return ENTREE_COTES_SCORES

    # Génération des features et prédiction
    features = generer_features(
        context.user_data["cotes"],
        context.user_data["cotes_scores"],
        context.user_data["cotes_double"]
    )
    classe, pourcentage = prediction_ensemble(features)

    # Prédiction pondérée des scores
    prediction_p1 = choix_score_pondere(context.user_data["scores"], context.user_data["cotes_scores"], classe)
    prediction_finale = choix_score_pondere(context.user_data["scores"], context.user_data["cotes_scores"], classe)
    top2_scores = scores_plus_probables(context.user_data["scores"], context.user_data["cotes_scores"])

    context.user_data["prediction"] = {
        "p1": prediction_p1,
        "finale": prediction_finale,
        "classe": classe,
        "pourcentage": pourcentage,
        "top2": top2_scores
    }

    keyboard = [
        [InlineKeyboardButton("✅ Gagnant", callback_data="gagnant")],
        [InlineKeyboardButton("❌ Perdant", callback_data="perdant")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"📊 Prédictions IA :\n"
        f"Période 1 : {prediction_p1}\n"
        f"Score final : {prediction_finale}\n"
        f"Prédiction générale : {classe} ({pourcentage}%)\n"
        f"Top 2 scores probables : {top2_scores[0][0]} ({top2_scores[0][1]}%), {top2_scores[1][0]} ({top2_scores[1][1]}%)\n\n"
        f"Ton pari a-t-il gagné ?",
        reply_markup=reply_markup
    )
    return RESULTAT

async def resultat(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    context.user_data["resultat_user"] = query.data
    if query.data == "gagnant":
        await query.edit_message_text("👍 Super ! Maintenant envoie le score exact réel du bookmaker.")
    else:
        await query.edit_message_text("😢 Dommage. Envoie quand même le score exact réel du bookmaker.")
    return SCORE_REEL

async def recevoir_score_reel(update: Update, context: CallbackContext):
    score_reel = update.message.text.strip()
    context.user_data["score_reel"] = score_reel

    global historique
    new_row = {
        "equipeA": context.user_data["equipeA"],
        "equipeB": context.user_data["equipeB"],
        "coteA": context.user_data["cotes"][0],
        "coteN": context.user_data["cotes"][1],
        "coteB": context.user_data["cotes"][2],
        "double1X": context.user_data["cotes_double"][0],
        "cote1X": context.user_data["cotes_double"][0],
        "double12": context.user_data["cotes_double"][1],
        "cote12": context.user_data["cotes_double"][1],
        "doubleX2": context.user_data["cotes_double"][2],
        "coteX2": context.user_data["cotes_double"][2],
        **{f"score{i+1}": context.user_data["scores"][i] for i in range(15)},
        **{f"cote_score{i+1}": context.user_data["cotes_scores"][i] for i in range(15)},
        "diff_coteA_B": context.user_data["cotes"][0] - context.user_data["cotes"][2],
        "ratio_double1X_X2": context.user_data["cotes_double"][0] / (context.user_data["cotes_double"][2]+0.01),
        "prediction_p1": context.user_data["prediction"]["p1"],
        "prediction_finale": context.user_data["prediction"]["finale"],
        "resultat": context.user_data["resultat_user"],
        "score_reel": score_reel
    }

    historique = pd.concat([historique, pd.DataFrame([new_row])], ignore_index=True)
    historique.to_csv(DATA_FILE, index=False)
    entrainer_models()
    joblib.dump(models, "models.pkl")

    await update.message.reply_text(f"✅ Merci ! Score réel enregistré ({score_reel}). L’IA s’améliore avec tes retours 🙌")
    return ConversationHandler.END

# ======================
# Commandes /stats et /historique
# ======================
async def stats(update: Update, context: CallbackContext):
    if historique.empty:
        await update.message.reply_text("⚠️ Aucun historique pour l’instant.")
        return
    texte_stats = f"📊 Statistiques IA (sur {len(historique)} matchs)\n\n"
    X = historique[["coteA","coteN","coteB","double1X","double12","doubleX2"] + [f"cote_score{i}" for i in range(1,16)] + ["diff_coteA_B","ratio_double1X_X2"]]
    y = historique["resultat"]

    for name, model in models.items():
        try:
            X_eval = X.copy()
            if name in scalers:
                X_eval = scalers[name].transform(X_eval)
            acc = model.score(X_eval, y)
            texte_stats += f"🤖 {name} : {acc*100:.1f}% réussite | poids ⚖️ {model_weights[name]:.2f}\n"
        except:
            texte_stats += f"🤖 {name} : Non entraîné\n"
    await update.message.reply_text(texte_stats)

async def afficher_historique(update: Update, context: CallbackContext):
    if historique.empty:
        await update.message.reply_text("⚠️ Aucun historique disponible.")
        return
    texte = "📋 Contenu de l'historique :\n"
    texte += historique.tail(10).to_string(index=False)
    await update.message.reply_text(texte)

# ======================
# Main
# ======================
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ENTREE_EQUIPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, recevoir_equipes)],
            ENTREE_COTES: [MessageHandler(filters.TEXT & ~filters.COMMAND, recevoir_cotes)],
            ENTREE_DOUBLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, recevoir_double)],
            ENTREE_SCORES: [MessageHandler(filters.TEXT & ~filters.COMMAND, recevoir_scores)],
            ENTREE_COTES_SCORES: [MessageHandler(filters.TEXT & ~filters.COMMAND, recevoir_cotes_scores)],
            RESULTAT: [CallbackQueryHandler(resultat)],
            SCORE_REEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, recevoir_score_reel)]
        },
        fallbacks=[]
    )

    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("historique", afficher_historique))
    app.run_polling()  # Render maintient le script actif tant que le service tourne

if __name__ == "__main__":
    main()
