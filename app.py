import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="LoanPredict AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    # 🏦 LoanPredict AI
    ### Smart Loan Decisions
    """,
    unsafe_allow_html=False
)

st.caption("AI-Powered Loan Approval Prediction System")

st.divider()


DATASET_FILE = "loan_approval_dataset.csv"
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"


def train_model():

    if not os.path.exists(DATASET_FILE):

        st.error(
            "loan_approval_dataset.csv was not found."
        )

        return None, None

    data = pd.read_csv(DATASET_FILE)

    data.columns = data.columns.str.strip()

    for column in data.select_dtypes(include="object").columns:
        data[column] = (
            data[column]
            .astype(str)
            .str.strip()
        )


    data["education"] = data["education"].map({
        "Graduate": 1,
        "Not Graduate": 0,
        "Graduated": 1,
        "Not Graduated": 0
    })

    data["self_employed"] = data["self_employed"].map({
        "Yes": 1,
        "No": 0
    })


    data["loan_status"] = data["loan_status"].map({
        "Approved": 1,
        "Rejected": 0
    })


    asset_columns = [
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ]

    available_assets = []

    for column in asset_columns:

        if column in data.columns:
            available_assets.append(column)

    if len(available_assets) > 0:

        data["total_assets"] = data[
            available_assets
        ].sum(axis=1)

    else:

        data["total_assets"] = 0


    features = [
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "total_assets"
    ]

    data = data.dropna(
        subset=features + ["loan_status"]
    )

    X = data[features]
    y = data["loan_status"]



    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)


    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_scaled,
        y
    )

    return model, scaler



@st.cache_resource
def load_model():

    if (
        os.path.exists(MODEL_FILE)
        and os.path.exists(SCALER_FILE)
    ):

        try:

            with open(
                MODEL_FILE,
                "rb"
            ) as file:

                model = pickle.load(file)

            with open(
                SCALER_FILE,
                "rb"
            ) as file:

                scaler = pickle.load(file)

            return model, scaler

        except Exception:

            pass

    model, scaler = train_model()

    if model is not None:

        with open(
            MODEL_FILE,
            "wb"
        ) as file:

            pickle.dump(
                model,
                file
            )

        with open(
            SCALER_FILE,
            "wb"
        ) as file:

            pickle.dump(
                scaler,
                file
            )

    return model, scaler


model, scaler = load_model()

if "page" not in st.session_state:

    st.session_state.page = "Home"


if "prediction" not in st.session_state:

    st.session_state.prediction = None


if "confidence" not in st.session_state:

    st.session_state.confidence = 0


nav1, nav2, nav3, nav4, nav5 = st.columns(
    [3, 1, 1.5, 1, 1]
)


with nav1:

    st.subheader("🏦 LoanPredict AI")
    st.caption("Smart Loan Decisions")


with nav2:

    if st.button(
        "Home",
        use_container_width=True
    ):

        st.session_state.page = "Home"

        st.session_state.prediction = None

        st.rerun()


with nav3:

    if st.button(
        "Loan Prediction",
        use_container_width=True
    ):

        st.session_state.page = "Prediction"

        st.rerun()


with nav4:

    if st.button(
        "About",
        use_container_width=True
    ):

        st.session_state.page = "About"

        st.rerun()


with nav5:

    if st.button(
        "Contact",
        use_container_width=True
    ):

        st.session_state.page = "Contact"

        st.rerun()


st.divider()

if st.session_state.page == "Home":

    st.write("")

    st.info(
        "🤖 Machine Learning Powered"
    )

    st.title(
        "Loan Approval Prediction System"
    )

    st.write(
        """
        This application predicts whether a loan will be
        approved based on applicant details using a
        Machine Learning model.
        """
    )

    st.write(
        "Get instant, data-driven loan prediction results."
    )

    st.write("")

    if st.button(
        "📈 Check My Eligibility",
        type="primary",
        use_container_width=True
    ):

        st.session_state.page = "Prediction"

        st.rerun()

    st.write("")

    st.divider()

    st.subheader(
        "Why Use LoanPredict AI?"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.info(
            """
            🎯

            **Loan Prediction**

            Predict loan approval using
            Machine Learning.
            """
        )

    with c2:

        st.info(
            """
            📊

            **Data Driven**

            Uses applicant and financial
            information.
            """
        )

    with c3:

        st.info(
            """
            ⚡

            **Instant Result**

            Get results within seconds.
            """
        )

    with c4:

        st.info(
            """
            🤖

            **AI Powered**

            Machine Learning based
            prediction system.
            """
        )

elif st.session_state.page == "Prediction":

    st.title(
        "📝 Applicant Details"
    )

    st.caption(
        "Enter the applicant's personal and financial information."
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader(
            "👤 Personal Information"
        )

        dependents = st.slider(
            "Number of Dependents",
            min_value=0,
            max_value=5,
            value=1
        )

        education = st.selectbox(
            "Education Level",
            [
                "Graduated",
                "Not Graduated"
            ]
        )

        self_employed = st.selectbox(
            "Self Employed",
            [
                "No",
                "Yes"
            ]
        )

        income = st.slider(
            "Annual Income",
            min_value=100000,
            max_value=100000000,
            value=2100000,
            step=100000
        )

        st.write(
            "Annual Income"
        )

        st.metric(
            "Income",
            f"₹{income:,}"
        )

    with right:

        st.subheader(
            "💳 Loan & Financial Details"
        )

        loan_amount = st.slider(
            "Loan Amount",
            min_value=100000,
            max_value=500000000,
            value=10900000,
            step=100000
        )

        st.metric(
            "Loan Amount",
            f"₹{loan_amount:,}"
        )

        loan_term = st.slider(
            "Loan Duration",
            min_value=1,
            max_value=30,
            value=12
        )

        st.metric(
            "Duration",
            f"{loan_term} years"
        )

        cibil = st.slider(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=519
        )

        if cibil < 600:

            st.error(
                f"❌ Poor CIBIL Score: {cibil}"
            )

        elif cibil < 750:

            st.warning(
                f"⚠️ Fair CIBIL Score: {cibil}"
            )

        else:

            st.success(
                f"✅ Excellent CIBIL Score: {cibil}"
            )

        total_assets = st.slider(
            "Total Assets Value",
            min_value=0,
            max_value=1000000000,
            value=36000000,
            step=100000
        )

        st.metric(
            "Total Assets",
            f"₹{total_assets:,}"
        )

    st.divider()

    st.subheader(
        "📋 Application Summary"
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:

        st.metric(
            "Dependents",
            dependents
        )

    with s2:

        st.metric(
            "Education",
            education
        )

    with s3:

        st.metric(
            "Self Employed",
            self_employed
        )

    with s4:

        st.metric(
            "Annual Income",
            f"₹{income:,}"
        )


    s5, s6, s7, s8 = st.columns(4)

    with s5:

        st.metric(
            "Loan Amount",
            f"₹{loan_amount:,}"
        )

    with s6:

        st.metric(
            "Loan Duration",
            f"{loan_term} years"
        )

    with s7:

        st.metric(
            "CIBIL Score",
            cibil
        )

    with s8:

        st.metric(
            "Total Assets",
            f"₹{total_assets:,}"
        )


    st.write("")

    if st.button(
        "🔍 Predict Loan Approval",
        type="primary",
        use_container_width=True
    ):

        if model is None or scaler is None:

            st.error(
                "Machine Learning model is not available."
            )

        else:

            if education == "Graduated":

                education_value = 1

            else:

                education_value = 0

            if self_employed == "Yes":

                employed_value = 1

            else:

                employed_value = 0

            input_data = np.array(
                [[
                    dependents,
                    education_value,
                    employed_value,
                    income,
                    loan_amount,
                    loan_term,
                    cibil,
                    total_assets
                ]]
            )


            scaled_data = scaler.transform(
                input_data
            )

            prediction = model.predict(
                scaled_data
            )[0]


            try:

                probability = model.predict_proba(
                    scaled_data
                )[0]

                confidence = (
                    max(probability) * 100
                )

            except Exception:

                confidence = 50


            st.session_state.prediction = int(
                prediction
            )

            st.session_state.confidence = confidence

            st.rerun()


    if st.session_state.prediction is not None:

        st.divider()

        result = st.session_state.prediction

        confidence = st.session_state.confidence


        if result == 1:

            st.success(
                "✅ LOAN APPROVED!"
            )

            st.subheader(
                "Congratulations!"
            )

            st.write(
                """
                Based on your profile, your loan is
                likely to be approved.
                """
            )

            st.metric(
                "Approval Confidence",
                f"{confidence:.0f}%"
            )

            st.progress(
                min(
                    int(confidence),
                    100
                )
            )

            st.write("")

            r1, r2, r3 = st.columns(3)

            with r1:

                st.info(
                    """
                    📄

                    **Gather Documents**

                    Gather all required
                    documents.
                    """
                )

            with r2:

                st.info(
                    """
                    🏦

                    **Visit Bank**

                    Visit your preferred bank.
                    """
                )

            with r3:

                st.info(
                    """
                    ✍️

                    **Complete Application**

                    Complete the formal
                    application.
                    """
                )

        else:

            st.error(
                "❌ LOAN REJECTED"
            )

            st.subheader(
                "Application needs improvement"
            )

            st.write(
                """
                Based on your current profile,
                your loan application may be rejected.
                Here's what you can improve:
                """
            )

            st.warning(
                "⚠️ Key Issues Found"
            )

            issue_found = False


            if cibil < 600:

                st.write(
                    "• Low CIBIL score (below 600) is a major risk factor."
                )

                issue_found = True


            if income < loan_amount:

                st.write(
                    "• Annual income is low relative to loan amount."
                )

                issue_found = True


            if total_assets < loan_amount:

                st.write(
                    "• Build more assets as collateral."
                )

                issue_found = True


            if not issue_found:

                st.write(
                    "• The applicant profile does not meet the model's approval pattern."
                )


            st.write("")

            r1, r2, r3 = st.columns(3)

            with r1:

                st.info(
                    """
                    📈

                    **Improve CIBIL**

                    Improve your CIBIL score.
                    """
                )

            with r2:

                st.info(
                    """
                    💰

                    **Improve Income**

                    Increase income or
                    reduce loan amount.
                    """
                )

            with r3:

                st.info(
                    """
                    🏦

                    **Build Assets**

                    Build more assets
                    as collateral.
                    """
                )


        st.write("")

        if st.button(
            "🔄 Try Another Prediction",
            use_container_width=True
        ):

            st.session_state.prediction = None

            st.rerun()


elif st.session_state.page == "About":

    st.title(
        "About LoanPredict AI"
    )

    st.write(
        """
        LoanPredict AI is an AI-powered Loan Approval
        Prediction System developed using Python,
        Machine Learning and Streamlit.

        The system accepts applicant information and
        financial details and predicts whether the
        loan application is likely to be approved
        or rejected.
        """
    )

    st.divider()

    st.subheader(
        "Technologies Used"
    )

    t1, t2, t3, t4 = st.columns(4)

    with t1:

        st.info(
            "🐍 Python"
        )

    with t2:

        st.info(
            "📊 Pandas"
        )

    with t3:

        st.info(
            "🤖 Machine Learning"
        )

    with t4:

        st.info(
            "🌐 Streamlit"
        )


elif st.session_state.page == "Contact":

    st.title(
        "Contact"
    )

    st.write(
        "Project: AI-Powered Loan Approval Prediction"
    )

    st.write(
        "Developer: R. Nikitha"
    )

    st.write(
        "Course: B.Sc. Computer Science with Artificial Intelligence"
    )

    st.write(
        "College: SDNB Vaishnav College"
    )


st.divider()

st.caption(
    "AI-Powered Loan Approval Prediction | Machine Learning Project"
)
