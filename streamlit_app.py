import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title('Ranking Based Conjoint Analyser')

    # Upload the Excel file
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("Uploaded Data:")
        st.write(df)
        
        # Treat the last column as the ranking column by default
        ranking_col = df.columns[-1]

        conjoint_attributes = [col for col in df.columns if col != ranking_col]
        
        # Create the model formula
        model_formula = f"{ranking_col} ~ " + " + ".join([f"C({attr}, Sum)" for attr in conjoint_attributes])
        
        # Fit the regression model
        model_fit = smf.ols(model_formula, data=df).fit()
        st.write("Regression Model Summary:")
        st.write(model_fit.summary())
        
        # Calculate part-worths and importance levels
        level_name = []
        part_worth = []
        part_worth_range = []
        important_levels = {}
        end = 1
        
        for item in conjoint_attributes:
            nlevels = len(list(np.unique(df[item])))
            level_name.append(list(np.unique(df[item])))
            begin = end
            end = begin + nlevels - 1
            new_part_worth = list(model_fit.params[begin:end])
            new_part_worth.append((-1) * sum(new_part_worth))
            important_levels[item] = np.argmax(new_part_worth)
            part_worth.append(new_part_worth)
            part_worth_range.append(max(new_part_worth) - min(new_part_worth))
        
        st.write("Part-Worths and Importance Levels:")
        st.write(f"Level names: {level_name}")
        st.write(f"Part-worths: {part_worth}")
        st.write(f"Part-worth range: {part_worth_range}")
        st.write(f"Important levels: {important_levels}")
        
        # Calculate attribute importance
        attribute_importance = [round(100 * (i / sum(part_worth_range)), 2) for i in part_worth_range]
        st.write(f"Attribute importance: {attribute_importance}")
        
        # Calculate the part-worths of each attribute level
        part_worth_dict = {}
        attrib_level = {}
        for item, i in zip(conjoint_attributes, range(0, len(conjoint_attributes))):
            for j in range(0, len(level_name[i])):
                part_worth_dict[level_name[i][j]] = part_worth[i][j]
                attrib_level[item] = level_name[i]
        
        st.write("Part-Worths of Each Attribute Level:")
        st.write(part_worth_dict)
        
        # Plot the relative importance of attributes
        plt.figure(figsize=(10, 5))
        sns.barplot(x=conjoint_attributes, y=attribute_importance)
        plt.title('Relative Importance of Attributes')
        plt.xlabel('Attributes')
        plt.ylabel('Importance')
        st.pyplot(plt)
        
        # Calculate the utility score for each profile
        utility = []
        for i in range(df.shape[0]):
            score = sum(part_worth_dict[df[attr][i]] for attr in conjoint_attributes)
            utility.append(score)
        df['utility'] = utility
        st.write("Utility Scores:")
        st.write(df[['utility']])
        
        # Display the profile with the highest utility score
        st.write("Profile with the Highest Utility Score:")
        st.write(df.iloc[np.argmax(utility)])
        
        # Display preferred levels in each attribute
        st.write("Preferred Levels in Each Attribute:")
        for item, j in zip(attrib_level.keys(), range(0, len(conjoint_attributes))):
            st.write(f"Preferred level in {item} is: {level_name[j][important_levels[item]]}")

if __name__ == "__main__":
    main()
