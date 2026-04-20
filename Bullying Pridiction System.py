
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("Bullying_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Bullying Prediction System", page_icon=":brain:", layout="wide")


#-----------------------------------------------------------------------------------------------------------------#

# Home Page 
def home_page():
    st.markdown('<div class="title"><h1>🏫 Bullying Victimization Risk Prediction System</h1>'
                '<div class="description">" Trained on a Global School-Based Student Health Survey (GSHS) Dataset in Argentina. | Dataset from Kaggle.com "</div></div>'
                , unsafe_allow_html=True)
    
    st.markdown('<div class="custom-image">.</div></div>'
                , unsafe_allow_html=True)

    with st.expander("**🧠 Understanding Bullying**"):
        st.info("""
        Bullying is an aggressive behavior that involves unwanted, negative actions.
        It usually involves a power imbalance and is repeated over time. \n\n
        Bullying can be:
        - **Physically**:  Hitting, Pushing
        - **Verbally**: Name-calling, Threats
        - **Social/Relational**: Exclusion, Spreading rumors
        - **Cyberbullying**: via social media, Texting \n\n
        The victims of bullying often experience anxiety, depression, low self-esteem, and may avoid school.
        """)


    with st.expander("**🔧 This system is to:**"):
        col1, col2 = st.columns(2)
        with col1:
            st.success("#### 📥 Prediction\nThis system is to predict whether a student are on risk to being bullid.")
        with col2:
            st.success("#### 🧠 Recommendation\nStudents, parents and educators can get recommendation on preventing bullying.")


    with st.expander(" **👥 Who Can Use This System?**"):
            st.success("""
            - 🏫 **Students**: Identify risk level by applying their informations.
            - 👨‍👩‍👧‍👦 **Parents**: Understand emotional and behavioral risk factors of their children.           
            - 🏫 **Educators**: Identify at-risk students early in school environment.
            """)


    with st.expander("**📈 Machine Learning Model Performance Overview**"):
        st.info("""
        **XGBoost Classifier**
        - **Accuracy**: 84% 
        - **Precision**: 85%  
        - **Recall**: 88%  
        - **F1 Score**: 86%
        """)

    st.markdown("> 🧠 *“Bullying is not a reflection of the victim’s character, but of the bully’s lack of it.”*")


### --- Home Page Interface Design --- ###
st.markdown("""
    <style>
    .title {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        height: 25vh;
        background-color: #e6f2ff; 
        color: darkblue;
        border-radius: 5px;
        max-width: 100% 
    }
    
    .title h1 {
        font-family: "Times New Roman", Times, serif;
        font-size: 45px;
        font-weight: bold;
    }

    .description {
        font-family: "Times New Roman", Times, serif;
        font-size: 12px;
        font-weight: 600;
        color: #4B0082;
        background-color: white;
        border-radius: 3px;
    }   
            
    .custom-image {
        background-image: url("https://familytutor.sg/wp-content/uploads/2020/07/tt_s2-768x432-1.jpg");
        background-position: center;
        height: 50vh;
        width: 100%;  
        margin-top: 10px;  
        margin-bottom: 20px;  
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


#-----------------------------------------------------------------------------------------------------------------#

# Map categorical data to numerical values
def map_categorical_data(input_data):
    physically_attacked_mapping = {
        "0 times": 0,
        "1 time": 1,
        "2-3 times": 2,
        "4-5 times": 4,
        "6-7 times": 6,
        "8-9 times": 8,
        "10-11 times": 10,
        "12 or more times": 12
    }

    physical_fighting_mapping = {
        "0 times": 0,
        "1 time": 1,
        "2-3 times": 2,
        "4-5 times": 4,
        "6-7 times": 6,
        "8-9 times": 8,
        "10-11 times": 10,
        "12 or more times": 12
    }

    missed_school_mapping = {
        "0 days": 0,
        "1-2 days": 1,
        "3-5 days": 3,
        "6-9 days": 6,
        "10 or more days": 10
    }

    close_friends_mapping = {
        "3 or more": 3,
    }

    # Apply the mappings
    input_data["Physically_attacked"] = input_data["Physically_attacked"].map(physically_attacked_mapping)
    input_data["Physical_fighting"] = input_data["Physical_fighting"].map(physical_fighting_mapping)
    input_data["Close_friends"] = input_data["Close_friends"].map(close_friends_mapping)
    input_data["Missed_school"] = input_data["Missed_school"].map(missed_school_mapping)


    return input_data


### --- Predict Page Interface Design --- ###
st.markdown("""
    <style>
    .headerpp {
        # background-color: #e9ffbe;  
        color: 	#0c2c4c;  
        padding: 20px;
        font-family: "Times New Roman", Times, serif;
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        border-radius: 5px;
    }
    .descriptionpp {
        font-size: 15px;
        font-weight: 500;
        color: #555555;
        text-align: center;
        # margin-top: 10px;
        margin-bottom: 30px;
        font-family: "Times New Roman", Times, serif;  
    }
            
    /* Style for selectbox background color */
    div[data-baseweb="select"] > div {
        background-color: #04342a;         */dark green*/
        border-radius: 5px;
    }

    /* Style the selected option */
    div[data-baseweb="select"] > div > div {
        color: white;  
    }
            
    .stSlider, .stSelectbox {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 15px;
        colour: white;
        background-color: #f0fee0;      /*light green*/
        border: 1px solid #79f038; 
    }
    .stButton>button {
        background-color: white;
        color: orange;
        padding: 12px 20px;
        border: 2px solid orange;
        border-radius: 8px;
        margin-bottom: 30px;
    }
            
    /* Change color when hovering over the button */
    .stButton>button:hover {
        background-color: orange;  
        color: white
    }
            
    .custom-space {
        height: 20px;  /* Adjust the space size as needed */
    }
    </style>
""", unsafe_allow_html=True)


# Prediction Page 
def prediction_page():
    st.markdown('<div class="headerpp">🗃️ Bullying Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="descriptionpp">Predict the bullying risk level of students and receive personalized recommendations from different aspects.</div>', unsafe_allow_html=True)

    # Role Selection
    role = st.selectbox("**Please select your role:**", ["Student", "Parent", "Educator"])
    
    # Collecting information based on the selected role
    st.subheader("**📋 Student's details:**")


    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("**Gender**", ["Male", "Female"])
        gender = 1 if gender == "Male" else 0
        age = st.slider("**Age**", 10, 18, 13)
        peer_helpfulness = st.slider("**Peer Helpfulness (1 = Never to 5 = Always)**", 1, 5, 3)
        physically_attacked = st.selectbox("**Physically Attacked (times)**", ["0 times", "1 time", "2-3 times", "4-5 times", "6-7 times", 
                                                                           "8-9 times", "10-11 times", "12 or more times"])
        close_friends = st.selectbox("**Number of Close Friends**", [0, 1, 2, "3 or more"])
        
    with col2:
        missed_school = st.selectbox("**Missed School (days)**", [0, "1-2 days", "3-5 days", "6-9 days", "10 or more days"])
        felt_lonely = st.slider("**Felt Lonely (1 = Never to 5 = Always)**", 1, 5, 3)
        parental_understanding = st.slider("**Parental Understanding (1 = Never to 5 = Always)**", 1, 5, 3)
        physical_fighting = st.selectbox("**Physical Fighting (times)**",["0 times", "1 time", "2-3 times", "4-5 times", "6-7 times", 
                                                                      "8-9 times", "10-11 times", "12 or more times"])
        weight_status = st.selectbox(
            "**Physical Appearence:**",
            ["Underweight", "Overweight", "Obese"]
        )

        were_underweight = 1 if weight_status == "Underweight" else 0
        were_overweight = 1 if weight_status == "Overweight" else 0
        were_obese = 1 if weight_status == "Obese" else 0


    # Predict button
    if st.button("Predict Bullying Risk"):
        input_data = pd.DataFrame([[
            age, gender, physically_attacked, physical_fighting,
            felt_lonely, close_friends, missed_school,
            peer_helpfulness, parental_understanding,
            were_underweight, were_overweight, were_obese, 0
        ]], columns=[
            "Age", "Gender", "Physically_attacked", "Physical_fighting",
            "Felt_lonely", "Close_friends", "Missed_school",
            "Peer_helpfulness", "Parental_understanding",
            "Were_underweight", "Were_overweight", "Were_obese", "Has_Outlier"
        ])

        # Map categorical data to numeric values
        input_data = map_categorical_data(input_data)

        prediction_proba = model.predict_proba(input_data)[0][1]  # prob of being bullied
        
        # Based on role selection, provide customized recommendations
        if role == "Student":
            if prediction_proba >= 0.65:
                st.subheader("Prediction Result:")
                st.error("**🚨 High Risk of Being Bullied**\n" \
                f"\n**📊 Probability of being bullied: `{prediction_proba * 100:.2f}%`**")
                
                st.markdown('<div class="custom-space"></div>', unsafe_allow_html=True)
                
                st.subheader("Recommendations:")
                st.write("It is crucial to intervene immediately. Seek support from a counselor, teacher, or trusted adult.")
                with st.expander("**🧠 What you can do?**"):
                    st.info("""
                    - Talk to a trusted adult such as a teacher, counselor, or parent.
                    - Avoid isolating yourself. Stay around supportive friends.
                    - Document bullying incidents (date, time, who was involved).
                    - Call helplines if needed.
                    """)
                with st.expander("**🛡️ How to Prevent Future Bullying?**"):
                    st.info("""
                    - Engage in activities that build self-esteem, like sports, hobbies, or volunteering.
                    - Bullying often occurs when individuals are isolated. Stay with friends or peers when possible.
                    - Learn assertive communication techniques to stand up for yourself respectfully.
                    """)
            
            elif 0.30 <= prediction_proba < 0.65:
                st.subheader("Prediction Result:")
                st.warning("**⚠️ Medium Risk of Being Bullied**\n" \
                f"\n**📊 Probability of being bullied: `{prediction_proba * 100:.2f}%`**")
                
                st.markdown('<div class="custom-space"></div>', unsafe_allow_html=True)

                st.subheader("Recommendations:")
                st.write("While the situation is not urgent, regular check-ins and emotional support from peers and parents could help.")
                with st.expander("**🧠 What you can do:**"):
                    st.info("""
                    - Talk to someone you trust about how you feel.
                    - Avoid being alone in places where bullying may occur.
                    - Seek guidance from school counselors or mentors.
                    """)
                with st.expander("**🛡️ Strengthen Support System**"):
                    st.info("""
                    - Join school clubs or activities that promote inclusivity and mutual respect.
                    - Keep communicating with positive peers so that they will uplift you.
                    - Participate in workshops or programs that teach communication and coping strategies.
                    """)
            
            else:
                st.subheader("Prediction Result:")
                st.success("**✅ Low Risk of Being Bullied**\n" \
                f"\n**📊 Probability of being bullied: `{prediction_proba * 100:.2f}%`**")

                st.markdown('<div class="custom-space"></div>', unsafe_allow_html=True)

                st.subheader("Recommendations:")
                st.write("You are less likely to being bully, just maintain!!!")
                with st.expander("**🧠 What you can do:**"):
                    st.info("""
                    - Stay aware of your surroundings, especially in social situations. 
                    - Report any unusual behavior or bullying. This can help your friend who are being bullied. 
                    - Maintain positive peer relationships.
                    """)
                with st.expander("**🛡️ Keep a Positive Environment**"):
                    st.info("""
                    - Be inclusive and support peers.
                    - Be a role model to show kindness and inclusivity to others to help create a supportive environment.
                    - Participate in anti-bullying programs that may be available in your school.
                    """)


        elif role == "Parent":
            if prediction_proba >= 0.65:
                st.subheader("Prediction Result:")
                st.error("**🚨 High Risk of Being Bullied**\n" \
                f"\n**📊 Probability of being bullied: `{prediction_proba * 100:.2f}%`**")
                
                st.markdown('<div class="custom-space"></div>', unsafe_allow_html=True)
                
                st.subheader("Recommendations:")
                st.write("Ensure your child feels emotionally supported and work with school authorities to address the issue.")
                with st.expander("**🧠 What should a parents do?**"):
                    st.info("""
                    - Open a conversation with your child. Ask how they feel and encourage them to share their experiences.
                    - Inform the school about the situation so they can intervene immediately.
                    - Monitor your child's behavior and emotional. Look out for signs of distress such as withdrawal, anxiety, or reluctance to go to school.
                    """)
                with st.expander("**🛡️ Preventive steps:**"):
                    st.info("""
                    - **Promote a safe environment at home**: Encourage your child to express themselves openly.
                    - **Teach conflict resolution skills**: Help your child learn to handle situations calmly and assertively.
                    - **Support extracurricular activities**: Encourage your child to participate in clubs, sports, or other activities to build friendships.
                    """)
            
            elif 0.30 <= prediction_proba < 0.65:
                st.subheader("Prediction Result:")
                st.warning("**⚠️ Medium Risk of Being Bullied**\n" \
                f"\n**📊 Probability of being bullied: `{prediction_proba * 100:.2f}%`**")
                
                st.markdown('<div class="custom-space"></div>', unsafe_allow_html=True)
                
                st.subheader("Recommendations:")
                st.write("Keep the lines of communication open with your child and work together with the school to monitor behavior.")
                with st.expander("**🧠 What should a parents do?**"):
                    st.info("""
                    - Check in regularly with your child to understand how they are feeling and what’s happening at school.
                    - Collaborate with teachers and counselors to monitor the situation and ensure your child feels safe.
                    - Encourage them to participate in activities where they can excel and make friends.
                    """)
                with st.expander("**🏡 Prevention at home:**"):
                    st.info("""
                    - Talk openly about bullying and safety with your child so they understand what it is and how to react.
                    - Praise and support your child’s achievements oftenly.
                    - Teach your child how to set boundaries with peers in a healthy way.
                    """)
                            
            else:
                st.subheader("Prediction Result:")
                st.success("**✅ Low Risk of Being Bullied**\n" \
                f"\n**📊 Probability of being bullied: `{prediction_proba * 100:.2f}%`**")
                
                st.markdown('<div class="custom-space"></div>', unsafe_allow_html=True)
                
                st.subheader("Recommendations:")
                st.write("Encourage your child to continue developing positive relationships and seek help if needed.")
                with st.expander("**🧩 How to maintain:**"):
                    st.info("""
                    - Keep open communication with your child. Ask your child about their day regularly, and make sure they feel comfortable talking about any issues.
                    - Create an open line of communication with the school to ensure your child’s well-being.
                    - If your child starts showing signs of anxiety or depression, address it early.
                    """)
                with st.expander("**🏠 Reinforcement at Home**"):
                    st.info("""
                    - Encourage your child to engage in activities that make them feel good about themselves.
                    - Teach your child the importance of empathy and kindness towards others.
                    - Help your child to engage with a positive group of friends who look out for each other.
                    - Continue building confidence and social skills.
                    """)



        elif role == "Educator":
            if prediction_proba >= 0.65:
                st.subheader("Prediction Result:")
                st.error("**🚨 High Risk of Being Bullied**\n" \
                f"\n**📊 Probability of being bullied: `{prediction_proba * 100:.2f}%`**")
                
                st.markdown('<div class="custom-space"></div>', unsafe_allow_html=True)
                
                st.subheader("Recommendations:")
                st.write("Immediate intervention is necessary. Coordinate with counselors and parents to develop a safety plan for the student.")
                with st.expander("**🧠 What should an educator do?**"):
                    st.info("""
                    - Talk to the student in a safe environment to understand their experience.
                    - Provide one-on-one support for the student.
                    - Work with counselors, social workers, and parents to address the issue.
                    - Monitor the student closely. Ensure that they feel safe, both physically and emotionally, at school.
                    """)
                with st.expander("**🛡️ Ways to prevent and stop bullying:**"):
                    st.info("""
                    - Review and enforce anti-bullying policies at school.
                    - Encourage students to engage in group activities and foster an inclusive school culture.
                    - Ensure that all school staff are trained in recognizing bullying and knowing how to respond effectively.
                    """)
                
            elif 0.30 <= prediction_proba < 0.65:
                st.subheader("Prediction Result:")
                st.warning("**⚠️ Medium Risk of Being Bullied**\n" \
                f"\n**📊 Probability of being bullied: `{prediction_proba * 100:.2f}%`**")
                
                st.markdown('<div class="custom-space"></div>', unsafe_allow_html=True)
                
                st.subheader("Recommendations:")
                st.write("Monitor the student's interactions closely and provide additional support through peer counseling or school resources.")
                with st.expander("**🧠 What should an educator do?**"):
                    st.info("""
                    - **Observe classroom dynamics**: Be vigilant in noticing any signs of bullying or social exclusion.
                    - **Talk to the student**: Engage them in a conversation to understand their experience and offer support.
                    - **Provide support groups**: Help the student connect with peers who can provide emotional support.
                    """)
                with st.expander("**🛡️ Preventive Actions:**"):
                    st.info("""
                    - **Teach empathy**: Include emotional intelligence training as part of the curriculum.
                    - **Provide peer mentoring**: Pair students with positive role models who can help them navigate social challenges.
                    - **Regular check-ins**: Keep monitoring the student’s well-being to ensure they are coping with the school environment.
                    """)
                
            else:
                st.subheader("Prediction Result: ")
                st.success("**✅ Low Risk of Being Bullied**\n" \
                f"\n**📊 Probability of being bullied: `{prediction_proba * 100:.2f}%`**")
                
                st.markdown('<div class="custom-space"></div>', unsafe_allow_html=True)
                
                st.subheader("Recommendations:")
                st.write("Continue to monitor the student's social dynamics and offer support as needed.")
                with st.expander("**🧠 What educator can do?**"):
                    st.info("""
                    - **Check in with the student regularly**: Make sure they feel comfortable sharing any concerns they might have.
                    - **Foster a positive school environment**: Encourage all students to engage in an inclusive and respectful manner.
                    - **Maintain Safe Environment**: Ensure the student knows they have a safe space at school to talk about their concerns.
                    """)
                with st.expander("**🛡️ Preventive steps:**"):
                    st.info("""
                    - **Promote empathy**: Continue fostering an environment of kindness and understanding.
                    - **Encourage teamwork**: Have students collaborate on projects and group work to promote positive socialization.
                    - **Celebrate diversity**: Emphasize the importance of embracing differences in your classroom.
                    - Watch for early signs of peer conflict.
                    """)
                    
    
        # Display helplines
        st.markdown('<div class="custom-space"></div>', unsafe_allow_html=True)
        st.subheader("Need any help?")
        with st.expander("**Call to Helpline**"):
            st.subheader("**Emergency Contacts and Helplines (Malaysia)**")
            st.error("""
            - **Talian Kasih (24/7 Hotline)**: 📞 15999  
            - **Befrienders KL** (emotional support): 📞 03-7627 2929 / 🌐 [befrienders.org.my](https://www.befrienders.org.my)
            - **Childline Malaysia** (for youth under 18): 📞 15999

            _You can call anonymously. Everything you say is confidential._
            """)


#-----------------------------------------------------------------------------------------------------------------#

# Sidebar Navigation
st.sidebar.title("Page Navigation")
tabs = ["🏠 Home", "🎯 Prediction"]
page = st.sidebar.radio("Select a Page", tabs)

if page == "🏠 Home":
    home_page()
elif page == "🎯 Prediction":
    prediction_page()

#-----------------------------------------------------------------------------------------------------------------#


### --- Additional Design --- ###
# Footer: Add contact information or credits
st.markdown("""
    <style>
    .footer {
        background-color: #ededed;
        color: gray;
        padding: 10px;
        text-align: center;
        font-size: 12px;
        position: fixed;
        bottom: 0;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Display footer
st.markdown('<div class="footer">🌎 Contact us at: support@bullyingpredictionsystem.com | &copy; YAP JIA XIN_TP066475 @ 2025 Bullying Prediction System</div>', unsafe_allow_html=True)



# streamlit run Bullying_predictive_system.py