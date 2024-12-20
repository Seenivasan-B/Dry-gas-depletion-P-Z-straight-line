#importing the essential libraries
import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from io import BytesIO
from io import StringIO

#Initialize the wide mode by default
st.set_page_config(layout="wide")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Adding a title 
st.title("Dry gas depletion P/Z straight line")

#Splitting the webpage into two columns in the ratio 1:2
col1, col2 = st.columns([1,2],gap="large")

# Initialize session state for decision and load state
if "decision" not in st.session_state:
    st.session_state.decision = "***Example Values***"
if "load_state" not in st.session_state:
    st.session_state.load_state = False

# Function to reset session state
# To reset all the values when the radio input got changed
def reset_state():
    for key in ["load_state", "dataframe", "final_df", "final_df1", "final_df2"]:
        if key in st.session_state:
            del st.session_state[key]

#Defining the radio button for various inputs | Placed at the left column
with col1:
        st.subheader("Inputs")     
        decision = st.radio(
            "Input of Choice",
            ["***Example Values***","***Import Values***", "***Enter or Paste Values***",],on_change=reset_state)

#Processing and Visualizing the value 
with col2:
    #Example values input
    if decision == "***Example Values***":
            gp = [0,4,5,6,7,8,12,19]
            p_avg = [5444,4957,4798,4671,4534,4426,4018,3406]
            dataframe = pd.DataFrame({'Gp(bscf)': gp, 'P_avg(psia)': p_avg})
            #The example values can be modified 
            edited_df = st.data_editor(dataframe,num_rows="dynamic",use_container_width=True)
            #Copying the dataframe for further calculations
            final_df  = edited_df.copy()
            final_df1 = edited_df.copy()
            final_df2 = edited_df.copy()
            st.success("Example data for reference.")

    #Import values input
    if decision == "***Import Values***":
            uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
            #Check if the uploaded file not null
            if uploaded_file is not None:
                # Determine file type and read accordingly
                if uploaded_file.name.endswith(".csv"):
                    dataframe = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    dataframe = pd.read_excel(uploaded_file)
                # Rename columns for consistency
                dataframe.columns = ['Gp(bscf)', 'P_avg(psia)']
                # Display the DataFrame
                edited_df = st.data_editor(dataframe,num_rows="dynamic",use_container_width=True)
                # Create copies of the DataFrame for further processing
                final_df  = edited_df.copy()
                final_df1 = edited_df.copy()
                final_df2 = edited_df.copy()
                # Display confirmation of processing
                st.success("File processed successfully!")
            else:
                #if no upload or upload file is none show this message
                st.info("Please upload a CSV or Excel file.")
    
    #Enter values input
    if decision == "***Enter or Paste Values***":
        # Text area for data input
        data_input = st.text_area(
            "Enter data as two columns (separated by spaces or tabs). Each row should represent a new data point.",
            placeholder="Example:\n0 5444\n4 4957\n5 4798"
        )
        if data_input:
            try:
                # Process the input data
                data_lines = data_input.strip().split("\n")  # Split input into lines
                data = [line.split() for line in data_lines]  # Split each line by spaces or tabs
                
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=['Gp(bscf)', 'P_avg(psia)'])
                df = df.replace(',', '', regex=True)
                df = df.astype(float)
                # Check for exactly 2 columns
                if df.shape[1] == 2:
                    st.success("Data processed successfully!")
                    # Display and allow editing
                    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
                    final_df = edited_df.copy()
                    final_df1 = edited_df.copy()
                    final_df2 = edited_df.copy()
                else:
                    st.warning("Input data must contain exactly 2 columns. Please check the format.")
            
            except Exception as e:
                st.error("Failed to process input data. Ensure proper formatting.")
                st.error(f"Error details: {e}")
        else:
            st.info("Waiting for you to enter data / Manually input the data in the table below...")
            # Placeholder table for manual entry
            empty_df = pd.DataFrame({'Gp(bscf)': [], 'P_avg(psia)': []})
            edited_df = st.data_editor(empty_df, num_rows="dynamic", use_container_width=True)
            final_df = edited_df.copy()
            final_df1 = edited_df.copy()
            final_df2 = edited_df.copy()

#--------------------------------------------------------------------------------------------------------------------------------------------------------#

#Creating two tabs from single line and dual line
tab1, tab2= st.tabs(["Single Line", "Dual Line"])

# Tab 1 Single line 
with tab1:
    #creating 2 columns for input and graph
    col3, col4 = st.columns([1,2],gap="large")
    with col3:
            #Finding the values for upper limit and lower limit of initial reservoir pressure
            try:
                uplim = None
                if round(final_df['P_avg(psia)'].max(),-3) > final_df['P_avg(psia)'].max():
                    uplim= round(final_df['P_avg(psia)'].max(),-3) 
                if round(final_df['P_avg(psia)'].max(),-3) < final_df['P_avg(psia)'].max():
                    uplim = round(final_df['P_avg(psia)'].max(),-3) + 1000
            except:
                print("Problem with caculation for initial reservoir pressure")
            #Finding the values for upper limit and lower limit of OGIP
            try:
                X = final_df['P_avg(psia)'].values.reshape(-1, 1)  # Feature (Pressure)
                y = final_df['Gp(bscf)'].values
                model = LinearRegression()
                model.fit(X, y) 
                pressure_zero = np.array([[0]])
                ogip_lim = model.predict(pressure_zero)[0] + (0.3*model.predict(pressure_zero)[0])
                if ogip_lim > 0:
                    ogiplim = abs(ogip_lim)
                else:
                    ogip_lim = 0
            except:
                print("Problem with calculation for OGIP")
            
            #Getting the inputs from user with the use of slider
            st.subheader("P/Z Analysis")
            try :
                pres_in = st.slider(label='Initial Reservoir Pressure (Psia)',min_value=float(uplim*0.7), max_value=float(uplim*1.3),step=1.,format='%.0f',value=float(uplim))
                res_temp = st.slider(label='Reservoir Temperature (°F)',min_value=60., max_value=250.,step=1.,format='%.0f',value = 100.0)
                gas_sg = st.slider(label='Gas Specific Gravity',min_value=0.5, max_value=.7,step=0.01,format='%0.2f',value=0.6)
                ogip = st.slider(label='OGIP(Bscf)',min_value=float(math.ceil((ogip_lim*0.5))), max_value=float(math.ceil((ogip_lim*2.5))),step=.1,format='%.1f',value=ogip_lim)
                pres_ab = st.slider(label='Abandonment Pressure (Psia)',min_value=100., max_value=(uplim*0.25),step=1.,format='%.0f',value=float(uplim/10))
            except:
                print("Problem with the input values of the slider")
            st.markdown("---") 
            #Changing the inital reservoir pressure from user input
            try:
                final_df['P_avg(psia)'][0] = pres_in
            except:
                print("Unable to fix pressure values")  

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Calculating of the gas compresssibility
            def gas_compressibility(temp,sg,pres):
                tpc = 169 + 314*sg
                ppc = 708.75 - (57.7*sg)
                tpr = (temp + 460) / tpc
                t = 1/tpr
                ppr = pres / ppc
                
                a = 0.06125*t* math.exp(-1.2*(1-t)**2)
                b = (14.76*t) - (9.76*t**2) + (4.58*t**3)
                c = (90.7*t) - (242.2*t**2) + (42.4*t**3)
                d = 2.18 + (2.82*t)

                Y = 0.0125 * ppr * t * math.exp(-1.2*(1-t)**2)
                
                def func(Y):
                    return -a*ppr + (Y+Y**2+Y**3-Y**4)/(1-Y)**3 - b*Y**2 + c*Y**d
                
                def derivFunc(Y):
                    return ((1+4*Y+4*Y**2-4*Y**3+Y**4)/(1-Y)**4) - 2*b*Y + c*d*Y**(d-1) 
                
                def newtonRaphson(Y):
                    h = func(Y) / derivFunc(Y)
                    while abs(h) >= 1e-15:
                        h = func(Y)/derivFunc(Y)
                        Y = Y - h
                    return Y
                
                Y = newtonRaphson(Y)
                
                return a*ppr / Y

#Making the final dataframe
            def zcalc(final_df,temp,sg):
                final_df['Z'] = final_df['P_avg(psia)'].apply(lambda x: gas_compressibility(temp,sg,x))
                final_df['P/Z'] = final_df['P_avg(psia)'] / final_df['Z']
                return final_df
            try:
                out = zcalc(final_df,res_temp,gas_sg)
            except:
                print("Problem with the calculation of Z values")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Visualizing the summary column
with col4:
        #calculation part     
        try :
            z_ab = gas_compressibility(res_temp,gas_sg,pres_ab)
            pz_ab = pres_ab / z_ab
            eur = (ogip / final_df['P/Z'][0]) * (final_df['P/Z'][0] - pz_ab)
            rf = (eur  / ogip ) * 100   
        except:
            print("Some errors in summary calculation")
        #Visualization part
        try:
            st.subheader( "Summary")   
            summary = pd.DataFrame({'OGIP(bscf)': ogip, 'EUR(bscf)': round(eur,1), 'RF(%)':round(rf,1)},index=[0])
            st.dataframe(summary,hide_index=True,use_container_width=False,width=600)
        except:
            print("Unable to view the summary")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Visualization of the plot
        try:
            st.subheader("Plot")
            fig = px.scatter(final_df[1:], y="P/Z", x="Gp(bscf)",labels = {
                "Gas in Place" : "Gp(bscf)"
            })
            
            #controlling dimensions
            fig.update_layout(
            width=1200, 
            height=600,
            )

            #Changing marker size
            fig.update_traces(
            marker=dict(size=10, symbol="diamond",color='#f0c571'),
            selector=dict(mode="markers"))

            #the grey line
            fig.add_shape(
                type="line",
                x0=0, y0=final_df['P/Z'][0], x1=ogip, y1=0,
                line=dict(color="black", width=1))

            #Adding the markers
            fig.add_scatter(x=[0],
                            y=[final_df['P/Z'][0]],
                            marker=dict(
                                color='#007191',
                                size=12),
                            name="P/Z@Pi" )
            fig.add_scatter(x=[0],
                            y=[pz_ab],
                            marker=dict(
                                color='#a559aa',
                                size=12),
                            name="P/Z@Pab" )
            fig.add_scatter(x=[ogip],
                            y=[0],
                            marker=dict(
                                color='#c9040f',
                                size=12),
                            name="OGIP" )
            fig.add_scatter(x=[eur],
                            y=[0],
                            marker=dict(
                                color='#59a89c',
                                size=12),
                            name="EUR" )
            #Horizontal dash line
            fig.add_shape(
            type="line",
            x0=0, 
            x1=eur,
            y0=pz_ab, 
            y1=pz_ab,
            line=dict(color="red", width=1, dash="dash"))

            #Vertical dash line
            fig.add_shape(
            type="line",
            x0=eur, 
            x1=eur,
            y0=pz_ab, 
            y1=0,
            line=dict(color="red", width=1, dash="dash"))

            #zero line
            fig.add_vline(x=0, line_width=1, line_color="grey")
            fig.add_hline(y=0, line_width=1, line_color="grey")

            st.plotly_chart(fig)
        except:
            print("Unable to generate graph")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Calcuation part of the download table

#Going back to column 3 from final analysis table
with col3:
        try:
            in_1 = [eur]
            in_2 = [pres_ab]
            new_df = pd.DataFrame({'Gp(bscf)': in_1, 'P_avg(psia)': in_2})
        except:
            print("Problem with calculation of the analysis table")
        
        try:
            new = zcalc(new_df,res_temp,gas_sg)
            def analysis(final_df,new,ogip):
                new_df = pd.concat([final_df,new],axis=0)
                new_df['Gp(bscf)'] = new_df['Gp(bscf)'].round(1)
                new_df['P_avg(psia)'] = new_df['P_avg(psia)'].astype(int)
                new_df['Z'] = new_df['Z'].round(4)
                new_df['P/Z'] = new_df['P/Z'].astype(int)
                new_df = new_df.reset_index(drop=True)
                new_df['Comment'] = new_df['Z']
                new_df['Comment'][0] = "Initial Pressure"
                new_df['Comment'][1:] = None
                new_df['Comment'][-1:] = "Abandonment(=>EUR)"
                new_df.loc[len(new_df)] = [round(ogip,1),None,None,0,"OGIP"]
                new_df.loc[len(new_df)] =['[bscf]','[psia]',None,None,None]
                new_df.columns = ['Gp','P_avg','Z','P/Z','Comment']
                last_row = new_df.iloc[-1]
                new_df = new_df.iloc[:-1]
                df = pd.concat([pd.DataFrame(last_row).T, new_df], ignore_index=True)
                return df
            export = analysis(final_df,new,ogip)
        except:
            print("Problem with the analysis table conversion")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Export part of the analysis table
        try:
            # Function to save DataFrames to an Excel file
            def save_to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name="Analysis", index=False)
                output.seek(0)
                return output

            # Generate the Excel file
            excel_file = save_to_excel(export)

            # Add a download button for the Excel file
            st.download_button(
                label="Download Analysis",
                data=excel_file,
                file_name="P_Z Analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
        except:
            print("Problem with exporting the table")

#-------------------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------End of single line section-------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------#

# Tab 2 dual line
with tab2:
    
    #Splitting the columns in the ratio 1:2 
     col5,col6 = st.columns([1,2],gap="large")
    
    #Getting inputs from the users
     with col5:
        try:
            st.subheader("P/Z Analysis-1")
            pres_in1 = st.slider(label='Initial Reservoir Pressure (Psia) ',min_value=float(uplim*0.7), max_value=float(uplim*1.3),step=1.,format='%.0f',value=pres_in)
            res_temp1 = st.slider(label='Reservoir Temperature (°F) ',min_value=60., max_value=250.,step=1.,format='%.0f',value = res_temp)
            gas_sg1 = st.slider(label='Gas Specific Gravity ',min_value=0.5, max_value=.7,step=0.01,format='%0.2f',value=gas_sg)
            ogip1 = st.slider(label='OGIP(Bscf) ',min_value=float(math.ceil((ogip_lim*0.5))), max_value=float(math.ceil((ogip_lim*2.5))),step=.1,format='%.1f',value=ogip)
            pres_ab1 = st.slider(label='Abandonment Pressure (Psia) ',min_value=100., max_value=(uplim*0.25),step=1.,format='%.0f',value=pres_ab)
        except:
            print("Problem with the slider input of analysis-1")
        st.markdown("---")
        try:
            st.subheader("P/Z Analysis-2")
            
            pres_in2 = st.slider(label='Initial Reservoir Pressure (Psia)  ',min_value=float(uplim*0.7), max_value=float(uplim*1.3),step=1.,format='%.0f',value=float(uplim))
            res_temp2 = st.slider(label='Reservoir Temperature (°F)  ',min_value=60., max_value=250.,step=1.,format='%.0f',value = 100.0)
            gas_sg2 = st.slider(label='Gas Specific Gravity  ',min_value=0.5, max_value=.7,step=0.01,format='%0.2f',value=0.6)
            ogip2 = st.slider(label='OGIP(Bscf)  ',min_value=float(math.ceil((ogip_lim*0.5))), max_value=float(math.ceil((ogip_lim*2.5))),step=.1,format='%.1f',value=ogip_lim)
            pres_ab2 = st.slider(label='Abandonment Pressure (Psia)  ',min_value=100., max_value=(uplim*0.25),step=1.,format='%.0f',value=float(uplim/10))
        except:
            print("Problem with the slider input of analysis-2")
        st.markdown("---")
        #Fixing the value of initial reservoir pressure from the slider input
        try:
            final_df1['P_avg(psia)'][0] = pres_in1
            final_df2['P_avg(psia)'][0] = pres_in2
            
        except:
            print("Problem with the inital reservoir pressure slider input") 

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Calculation part of dual line
        try:
            out1 = zcalc(final_df1,res_temp1,gas_sg1)
            out2 = zcalc(final_df2,res_temp2,gas_sg2)

        except:
            print("Problem with the calculation of Z values-Dual line")            

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Summary table calculation
with col6:
        #calculation part     
        try :
            z_ab1 = gas_compressibility(res_temp1,gas_sg1,pres_ab1)
            z_ab2 = gas_compressibility(res_temp2,gas_sg2,pres_ab2)
            pz_ab1 = pres_ab1 / z_ab1
            pz_ab2 = pres_ab2 / z_ab2
            eur1 = (ogip1 / final_df1['P/Z'][0]) * (final_df1['P/Z'][0] - pz_ab1)
            eur2 = (ogip2 / final_df2['P/Z'][0]) * (final_df2['P/Z'][0] - pz_ab2)
            rf1 = (eur1  / ogip1 ) * 100   
            rf2 = (eur2  / ogip2 ) * 100  
        except:
            print("Some errors in summary calculation- Dual line")
        
        #Visualization part
        try:
            st.subheader("Summary")
            summary1 = pd.DataFrame({"Analysis" : 1,'OGIP(bscf)': ogip1, 'EUR(bscf)': round(eur1,1), 'RF(%)':round(rf1,1)},index=[0])
            summary2 = pd.DataFrame({"Analysis" : 2,'OGIP(bscf)': ogip2, 'EUR(bscf)': round(eur2,1), 'RF(%)':round(rf2,1)},index=[0])
            final_summary = pd.concat([summary1,summary2],axis=0)
            st.dataframe(final_summary,hide_index=True,use_container_width=False,width=600)

        except:
            print("Problem with summary table visualization- Dual line")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Graph Visualization dual line
        try:
            st.subheader("Plot")
            #eur = round(eur,0)
            fig = px.scatter(final_df1[1:], y="P/Z", x="Gp(bscf)",labels = {
                "Gas in Place" : "Gp(bscf)"
            })
            #controlling dimensions
            fig.update_layout(
            width=1200, 
            height=600,
            )

            #Changing marker size
            fig.update_traces(
            marker=dict(size=10, symbol="diamond",color='#f0c571'),
            selector=dict(mode="markers"))

            #the grey line
            fig.add_shape(
                type="line",
                x0=0, y0=final_df1['P/Z'][0], x1=ogip1, y1=0,
                line=dict(color="black", width=1))
            
            fig.add_shape(
                type="line",
                x0=0, y0=final_df2['P/Z'][0], x1=ogip2, y1=0,
                line=dict(color="black", width=1,dash="dash"))

            #Adding the markers
            fig.add_scatter(x=[0],
                            y=[final_df1['P/Z'][0]],
                            marker=dict(
                                color='#007191',
                                size=12),
                            name="P/Z@Pi-1" )
            fig.add_scatter(x=[0],
                            y=[final_df2['P/Z'][0]],
                            marker=dict(
                                color='#00b9ed',
                                size=12),
                            name="P/Z@Pi-2" )
                            
            fig.add_scatter(x=[0],
                            y=[pz_ab1],
                            marker=dict(
                                color='#a559aa',
                                size=12),
                            name="P/Z@Pab-1" )
            fig.add_scatter(x=[0],
                            y=[pz_ab2],
                            marker=dict(
                                color='#f083f7',
                                size=12),
                            name="P/Z@Pab-2" )
            fig.add_scatter(x=[ogip1],
                            y=[0],
                            marker=dict(
                                color='#c9040f',
                                size=12),
                            name="OGIP-1" )
            fig.add_scatter(x=[ogip2],
                            y=[0],
                            marker=dict(
                                color='#ff0311',
                                size=12),
                            name="OGIP-2" )
            fig.add_scatter(x=[eur1],
                            y=[0],
                            marker=dict(
                                color='#59a89c',
                                size=12),
                            name="EUR-1" )
            fig.add_scatter(x=[eur2],
                            y=[0],
                            marker=dict(
                                color='#7febda',
                                size=12),
                            name="EUR-2" )
            #Horizontal dash line
            fig.add_shape(
            type="line",
            x0=0, 
            x1=eur1,
            y0=pz_ab1, 
            y1=pz_ab1,
            line=dict(color="red", width=1))
            fig.add_shape(
            type="line",
            x0=0, 
            x1=eur2,
            y0=pz_ab2, 
            y1=pz_ab2,
            line=dict(color="#6e0000", width=1, dash="dash"))

            #Vertical dash line
            fig.add_shape(
            type="line",
            x0=eur1, 
            x1=eur1,
            y0=pz_ab1, 
            y1=0,
            line=dict(color="red", width=1))
            fig.add_shape(
            type="line",
            x0=eur2, 
            x1=eur2,
            y0=pz_ab2, 
            y1=0,
            line=dict(color="#6e0000", width=1, dash="dash"))

            #zero line
            fig.add_vline(x=0, line_width=1, line_color="grey")
            fig.add_hline(y=0, line_width=1, line_color="grey")

            
        
            st.plotly_chart(fig)
            
        except:
            print("Provide proper input")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Calcuation part of the download table

#Going back to column 5 from final analysis table
        with col5:
             
             try:
                in_11 = [eur1]
                in_12 = [eur2]
                in_21 = [pres_ab1]
                in_22 = [pres_ab2]
                new_df1 = pd.DataFrame({'Gp(bscf)': in_11, 'P_avg(psia)': in_21})
                new_df2 = pd.DataFrame({'Gp(bscf)': in_12, 'P_avg(psia)': in_22})
             except:
                print("Problem with the calculation of analysis table dual line")
            
             try:
                new1 = zcalc(new_df1,res_temp1,gas_sg1)
                new2 = zcalc(new_df2,res_temp2,gas_sg2)
                export1 = analysis(final_df1,new1,ogip1)
                export2 = analysis(final_df2,new2,ogip2)
             except:
                print("Problem with the analysis table conversion Dual line")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Export part of the analysis table dual line
             try:
                # Function to save DataFrames to an Excel file
                def save_to_excel(df1, df2):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df1.to_excel(writer, sheet_name="Analysis_1", index=False)
                        df2.to_excel(writer, sheet_name="Analysis_2", index=False)
                    output.seek(0)
                    return output

                # Generate the Excel file
                excel_file = save_to_excel(export1, export2)

                # Add a download button for the Excel file
                st.download_button(
                    label="Download Analysis ",
                    data=excel_file,
                    file_name="P_Z Analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")    
             except:
                 print("Problem with the export table dual line")
#-------------------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------End of Dual line section-------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------#