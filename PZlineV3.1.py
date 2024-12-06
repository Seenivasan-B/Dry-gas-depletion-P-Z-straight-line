#importing the essential libraries
import streamlit as st
st.set_page_config(layout="wide")
import math
import pandas as pd
import plotly.express as px


#Adding a title 
st.title("Dry gas depletion P/Z straight line")
#Splitting the webpage into two columns
col1, col2 = st.columns([1,2],gap="large")

with col1:
        #Getting an input from user importing or modifying values
        st.subheader("Inputs")
        
        
        decision = st.radio(
            "Input of Choice",
            ["***Example Values***","***Import Values***", "***Enter Values***",]
        )

        if decision == "***Import Values***":
                #Import section
                uploaded_file = st.file_uploader("Choose a file")
                if uploaded_file is not None:
                        dataframe = pd.read_csv(uploaded_file)
                        dataframe.columns = ['Gp(bscf)','P_avg(psia)']
                        st.write(dataframe)
                        final_df = dataframe
                        
                        #final_df['P_avg(psia)'][0] = pres_in

        if decision == "***Example Values***":
                #Enter section
                gp = [0,4,5,6,7,8,12,19]
                p_avg = [5444,4957,4798,4671,4534,4426,4018,3406]
                dataframe = pd.DataFrame({'Gp(bscf)': gp, 'P_avg(psia)': p_avg})
                edited_df = st.data_editor(dataframe,num_rows="dynamic",use_container_width=True)
                final_df = edited_df 
        
        if decision == "***Enter Values***":
                #Enter section
                gp = []
                p_avg = []
                dataframe = pd.DataFrame({'Gp(bscf)': gp, 'P_avg(psia)': p_avg})
                edited_df = st.data_editor(dataframe,num_rows="dynamic",use_container_width=True)
                final_df = edited_df 
                
                #final_df['P_avg(psia)'][0] = pres_in          
        st.markdown("---")
        ## Fixing upper limit
        try:
            uplim = None
            if round(final_df['P_avg(psia)'].max(),-3) > final_df['P_avg(psia)'].max():
                uplim= round(final_df['P_avg(psia)'].max(),-3) 
            if round(final_df['P_avg(psia)'].max(),-3) < final_df['P_avg(psia)'].max():
                uplim = round(final_df['P_avg(psia)'].max(),-3) + 2000
        except:
             print("Provide Proper input")
#Section 2 end 
        st.subheader("P/Z Analysis")
        try :
            pres_in = st.slider(label='Initial Reservoir Pressure (Psia)',min_value=500., max_value=float(uplim),step=1.,format='%.0f',value=5444.)
            res_temp = st.slider(label='Reservoir Temperature (°F)',min_value=60., max_value=250.,step=1.,format='%.0f',value = 100.0)
            gas_sg = st.slider(label='Gas Specific Gravity',min_value=0.5, max_value=.7,step=0.01,format='%0.2f',value=0.6)
            ogip = st.slider(label='OGIP(Bscf)',min_value=0., max_value=500.,step=.1,format='%.1f',value=74.9)
            pres_ab = st.slider(label='Abandonment Pressure (Psia)',min_value=100., max_value=1000.,step=1.,format='%.0f',value=629.)
        except:
             print("Provide Proper input")
        st.markdown("---") 
        #Updating the inputs
        try:
            final_df['P_avg(psia)'][0] = pres_in
        except:
             print("Provide proper inputs")  

###Calculation part of Z
        def gas_compressibility(pres):
        
            temp = res_temp 
            
            sg = gas_sg
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

#Calculation part of z and P/Z       
        def zcalc():
              
            final_df['Z'] = final_df['P_avg(psia)'].apply(gas_compressibility)
            final_df['P/Z'] = final_df['P_avg(psia)'] / final_df['Z']
            return final_df

#Error handling for the name error        
        
        try:
            #st.subheader("To the dowload the caculated Z and P/Z values")
            download = zcalc()
            #downloading as a csv file
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode("utf-8")
            csv = convert_df(download)
            if len(final_df) == 0:
                 pass
            else:
                st.download_button(
                label="Download Analysis",
                data=csv,
                file_name="Calculated_Z.csv",
                mime="text/csv",
)
            #st.write(zcalc())
        except NameError:
            print("Import values to view the final table")

with col2:
    
#viewing the summary    
    try :
        
        z_ab = gas_compressibility(pres_ab)
        pz_ab = pres_ab / z_ab
        eur = (ogip / final_df['P/Z'][0]) * (final_df['P/Z'][0] - pz_ab)
        rf = (eur  / ogip ) * 100
        
        
        
        
        
    except:
         print("Provide necessary inputs")
    try:
        st.subheader( "Summary")   
        summary = pd.DataFrame({'OGIP(bscf)': ogip, 'EUR(bscf)': eur, 'RF(%)':rf},index=[0])
        st.dataframe(summary,hide_index=True,use_container_width=True)
    except:
         print("Provide necessary inputs")

#plotting the graph
    
    #Adding a scatterplot
    try:
        st.subheader("Plot")
        #eur = round(eur,0)
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
                            color='#e02b35',
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
        y0=pres_ab, 
        y1=0,
        line=dict(color="red", width=1, dash="dash"))

        
    
        st.plotly_chart(fig)
    except:
        print("Provide proper input")