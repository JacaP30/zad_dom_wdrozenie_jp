import json
import streamlit as st
import pandas as pd # type: ignore
from pycaret.clustering import load_model, predict_model # type: ignore
import plotly.express as px # type: ignore


# GŁÓWNA APLIKACJA 
#====================================================================
# Ta aplikacja pozwala znaleźć znajomych na podstawie podobieństw w danych demograficznych i zainteresowaniach.
# Wykorzystuje model klastrowania do grupowania uczestników.
# ====================================================================


MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'# <--- NAZWA MODELU

DATA = 'welcome_survey_simple_v2.csv' # <--- NAZWA PLIKU Z DANYMI

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json' # <--- NAZWA PLIKU Z NAZWAMI I OPISAMI KLASTRÓW


# Funkcje pomocnicze do ładowania modelu i danych
#====================================================================

@st.cache_data # Używamy cache_data do przechowywania modelu i danych
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r",) as f:
        return json.load(f)

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';') 
    df_with_clusters = predict_model(model, data=all_df) # Przewidujemy klastry dla wszystkich uczestników

    return df_with_clusters # Zwracamy DataFrame z klastrami

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    # tworzy kolejne zmienne na podstawie wyborów w sidebar
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([ # tworzy DataFrame z danymi osoby
        # Wartości są pobierane z sidebar
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])
#====================================================================
model = get_model() # Ładujemy model klastrowania
all_df = get_all_participants() # Pobieramy wszystkich uczestników z klastrami
cluster_names_and_descriptions = get_cluster_names_and_descriptions() # Pobieramy nazwy i opisy klastrów
predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0] # Przewidujemy ID klastra dla osoby
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id] # Pobieramy dane o przewidzianym klastrze

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

# Wyświetlamy dane o osobach z tego samego klastra
st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)
