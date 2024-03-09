from flask import Flask, jsonify, render_template
import plotly
import plotly.graph_objs as go
import json
from flask import Flask, jsonify
import plotly
import plotly.express as px
import json
from plotly.subplots import make_subplots
app = Flask(__name__)

import pandas as pd
import numpy as np

rawData = pd.read_excel("../rawData.xlsx")
print(rawData.columns)


rawData = rawData[[
                    "DisNo.",
                    "Historic",
                    "Classification Key",
                    "Disaster Group",
                    "Disaster Subgroup",
                    "Disaster Type",
                    "Disaster Subtype",
                    "ISO",
                    "Country",
                    "Subregion",
                    "Region",
                    "Location",
                    "AID Contribution ('000 US$)",
                    "Magnitude",
                    "Magnitude Scale",
                    "Latitude",
                    "Longitude",
                    "Start Year",
                    "Start Month",
                    "Start Day",
                    "End Year",
                    "End Month",
                    "End Day",
                    "Total Deaths",
                    "No. Injured",
                    "No. Affected",
                    "No. Homeless",
                    "Total Affected",
                    "Total Damage ('000 US$)",
                    "Total Damage, Adjusted ('000 US$)",
                    "CPI"                    
                    ]]

rawData.columns = [
                    "DisNo",
                    "Historic",
                    "Classification_Key",
                    "Disaster_Group",
                    "Disaster_Subgroup",
                    "Disaster_Type",
                    "Disaster_Subtype",
                    "ISO",
                    "Country",
                    "Subregion",
                    "Region",
                    "Location",
                    "AID_Contribution",
                    "Magnitude",
                    "Magnitude_Scale",
                    "Latitude",
                    "Longitude",
                    "Start_Year",
                    "Start_Month",
                    "Start_Day",
                    "End_Year",
                    "End_Month",
                    "End_Day",
                    "Total_Deaths",
                    "No_Injured",
                    "No_Affected",
                    "No_Homeless",
                    "Total_Affected",
                    "Total_Damage",
                    "Total_Damage_Adjusted",
                    "CPI"                    
                    ]

rawData["Start_Day"] = rawData["Start_Month"].fillna(1)
rawData["Start_Month"] = rawData["Start_Month"].fillna(1)

rawData["Start_Day"] = rawData["Start_Day"].astype(int).astype(str)
rawData["Start_Month"] = rawData["Start_Month"].astype(int).astype(str)
rawData["Start_Year"] = rawData["Start_Year"].astype(int).astype(str)


rawData["End_Day"] = rawData.apply(lambda row: 1 if np.isnan(row["End_Day"]) else row["End_Day"],axis = 1)
rawData["End_Month"] = rawData.apply(lambda row: 12 if np.isnan(row["End_Month"]) else row["End_Month"],axis = 1)
rawData["End_Day"] = rawData["End_Day"].astype(int).astype(str)
rawData["End_Month"] = rawData["End_Month"].astype(int).astype(str)
rawData["End_Year"] = rawData["End_Year"].astype(int).astype(str)
 
rawData["Start_Date"] = rawData.apply(lambda row: pd.to_datetime(f"{row['Start_Day'] + '/' + row['Start_Month'] + '/' + row['Start_Year']}", format="%d/%m/%Y"), axis = 1)
rawData["End_Date"] = rawData.apply(lambda row: pd.to_datetime(f"{row['End_Day'] + '/' + row['End_Month'] + '/' + row['End_Year']}", format="%d/%m/%Y"), axis = 1)

rawData["Start_Day"] = rawData["Start_Day"].astype(int)
rawData["Start_Month"] = rawData["Start_Month"].astype(int)
rawData["Start_Year"] = rawData["Start_Year"].astype(int)

# EDA

for elm in ["Disaster_Group", "Disaster_Subgroup", "Disaster_Type", "Disaster_Subtype"]:
    print(f"{elm}:", rawData[elm].unique(), end = "\n\n")

rawData = rawData.loc[rawData["Disaster_Group"] == "Natural"] # We only take 'Natural' disasters.
disCount = rawData.shape[0]
print("Number of natural disasters", disCount)

for elm in ["Disaster_Group", "Disaster_Subgroup", "Disaster_Type", "Disaster_Subtype"]:
    print(f"{elm}:", rawData[elm].unique(), end = "\n\n")


print("Number of countries in the dataset:", rawData['Country'].unique().shape[0], end = "\n\n")

print("Number of sub-regions in the dataset:", rawData['Subregion'].unique().shape[0])
print("Subregions:",  rawData['Subregion'].unique(), end = "\n\n")

print("Number of regions in the dataset:", rawData['Region'].unique().shape[0])
print("Subregions:",  rawData['Region'].unique(), end = "\n\n")


print("Date Range:", rawData["Start_Date"].min(), rawData["Start_Date"].max())

# BURADA HANGI KOLONU TARGET FEATURE OLARAK BELIRLIYORUZ ONU SECIYORUZ TOTAL_DAMAGE KOLONU YÜZDE 70 BOŞ OLDUĞU İÇİN DİĞER KOLONLARI KULLANCAĞIZ.
print("Percentage of None in Total_Damage", round(sum(rawData["Total_Damage"].isna()) / disCount, 2))
print("Percentage of None in Total_Affected", round(sum(rawData["Total_Affected"].isna()) / disCount, 2))
print("Percentage of None in Total_Deaths", round(sum(rawData["Total_Deaths"].isna()) / disCount, 2))

rawData["Decade"] = rawData['Start_Year'].astype(int)//10*10  


# BURASINDA SADECE DAMAGE'I İFADE EDEBİLDİĞİMİZ DISASTERLARI ALIYORUZ VE EXTRA-TERRESTRIALLARI DROPLUYORUM ÇÜNKÜ SAMPLE SIZE AZ.
filteredData = rawData.copy()
filteredData = filteredData.loc[~(filteredData["Total_Damage"].isna()) | (~filteredData["Total_Deaths"].isna())]
filteredData = filteredData.loc[filteredData["Disaster_Subgroup"] != "Extra-terrestrial"]


disasterMap = {}

for disasterGroup in filteredData["Disaster_Group"].unique():
    if disasterGroup not in disasterMap:
        disasterMap[disasterGroup] = {}
    for disasterSubgroup in filteredData[filteredData["Disaster_Group"] == disasterGroup]["Disaster_Subgroup"].unique():
        if disasterSubgroup not in disasterMap[disasterGroup]:
            disasterMap[disasterGroup][disasterSubgroup] = {}
        for disasterType in filteredData[(filteredData["Disaster_Group"] == disasterGroup) & (filteredData["Disaster_Subgroup"] == disasterSubgroup)]["Disaster_Type"].unique(): 
            if disasterType not in disasterMap[disasterGroup][disasterSubgroup]:
                disasterMap[disasterGroup][disasterSubgroup][disasterType] = []           
            for disasterSubType in filteredData[(filteredData["Disaster_Group"] == disasterGroup) & (filteredData["Disaster_Subgroup"] == disasterSubgroup) & (filteredData["Disaster_Type"] == disasterType)]["Disaster_Subtype"].unique(): 
                disasterMap[disasterGroup][disasterSubgroup][disasterType].append(disasterSubType)

# Using this map we can see unique disaster groups and types such that
print(disasterMap["Natural"]["Geophysical"]["Earthquake"])


decadeDeaths_ByDisaster_Group = filteredData.groupby(["Disaster_Group","Decade"])["Total_Deaths"].sum().reset_index()
decadeDeaths_ByDisaster_SubGroup = filteredData.groupby(["Disaster_Subgroup","Decade"])["Total_Deaths"].sum().reset_index()
decadeDeaths_ByDisaster_Type = filteredData.groupby(["Disaster_Type","Decade"])["Total_Deaths"].sum().reset_index()


yearlyDeaths_ByDisaster_Group = filteredData.groupby(["Disaster_Group","Start_Year"])["Total_Deaths"].sum().reset_index()
yearlyDeaths_ByDisaster_SubGroup = filteredData.groupby(["Disaster_Subgroup","Start_Year"])["Total_Deaths"].sum().reset_index()
yearlyDeaths_ByDisaster_Type = filteredData.groupby(["Disaster_Type","Start_Year"])["Total_Deaths"].sum().reset_index()
yearlyDeaths_ByDisaster_Type_Regional = filteredData.groupby(["Disaster_Type","Region","Decade"])["Total_Deaths"].sum().reset_index()


yearlyEvents = rawData.groupby(["Start_Year"])["DisNo"].count()
yearlyEvents = yearlyEvents.reset_index()
yearlyEvents_ByDisaster_Type = rawData.groupby(["Disaster_Type","Start_Year"])["DisNo"].count()
decadelEvents_ByDisaster_Type = rawData.groupby(["Disaster_Type","Decade"])["DisNo"].count()

yearlyEvents_ByDisaster_Type = yearlyEvents_ByDisaster_Type.reset_index()
decadelEvents_ByDisaster_Type = decadelEvents_ByDisaster_Type.reset_index()

yearlyRegionalEvents_ByDisasterGroup = filteredData.groupby(["Disaster_Group","ISO","Country"])["DisNo"].count().reset_index()
yearlyRegionalDeaths_ByDisasterGroup = filteredData.groupby(["Start_Year","Disaster_Group","ISO","Country"])["Total_Deaths"].sum().reset_index()
decadelRegionalEvents_ByDisasterGroup = filteredData.groupby(["Decade","Disaster_Group","ISO","Country"])["DisNo"].count().reset_index()
regionalDeaths_ByDisasterGroup =  filteredData.groupby(["Disaster_Group","ISO","Country"])["Total_Deaths"].sum().reset_index()
decadelRegionalDeaths_ByDisasterGroup =  filteredData.groupby(["Decade","Disaster_Group","ISO","Country"])["Total_Deaths"].sum().reset_index()

regionalDeaths_ByDisasterGroup["Total_Deaths"] = np.log(regionalDeaths_ByDisasterGroup["Total_Deaths"]) 

grouped_counts = filteredData.groupby(["Disaster_Group", "ISO", "Country", "Disaster_Type"]).size().reset_index(name='count')

sorted_counts = grouped_counts.sort_values(by=["Disaster_Type", "count"], ascending=False)

top_5_rows_per_type = sorted_counts.groupby("Disaster_Type").head(5)
subFilteredData = filteredData.loc[filteredData.Disaster_Type.isin(["Earthquake","Flood","Storm"])]

#################################### DEVELOPMENT PART ####################################
hdi = pd.read_csv("../hdr.csv")
melted_df = pd.melt(hdi, id_vars=['iso3', 'country', 'hdicode', 'region'], 
                    var_name='indicator', value_name='value')
melted_df["indicator"],melted_df["year"] = zip(*melted_df["indicator"].apply(lambda x: ("_".join(x.split("_")[:-1]), x.split("_")[-1])))
hdiFormatted = melted_df.pivot(index=['iso3',"country","hdicode","region","year"], columns="indicator", values='value').reset_index()
hdiFormatted.year = hdiFormatted.year.astype(int)
subFilteredData = pd.merge(subFilteredData, hdiFormatted, left_on=["ISO","Start_Year"], right_on=["iso3","year"], suffixes=('','_GDP'), how="left")
subFilteredData = subFilteredData.loc[subFilteredData.Start_Year > 1990]

floodDataAll = subFilteredData.loc[(subFilteredData["Disaster_Type"] =="Flood")]
floodData = subFilteredData.loc[(subFilteredData["Disaster_Type"] =="Flood") & (subFilteredData["ISO"].isin(["CHN","IND","IDN","BRA","USA"]))].copy()
stormDataAll = subFilteredData.loc[(subFilteredData["Disaster_Type"] =="Storm")]
stormData = subFilteredData.loc[(subFilteredData["Disaster_Type"] =="Storm") & (subFilteredData["ISO"].isin(["USA","PHL","CHN","IND","JPN"]))].copy()
earthquakeAll = subFilteredData.loc[(subFilteredData["Disaster_Type"] =="Earthquake")]
earthquake = subFilteredData.loc[(subFilteredData["Disaster_Type"] =="Earthquake") & (subFilteredData["ISO"].isin(["CHN","IDN","IRN","TUR","JPN"]))].copy()

groupedFloodData = floodDataAll.groupby(["ISO","Country"]).agg({"Total_Deaths":"sum", "gii": "mean", "gdi":"mean", "hdi":"mean", "le":"mean","gnipc":"mean","Disaster_Group":"count"}).reset_index()
groupedFloodData.columns = ["ISO","Country","Total_Deaths","gii","gdi","hdi","le","gnipc","Total_Event_Count"]
groupedFloodData["deathPerEvent"] = groupedFloodData["Total_Deaths"] / groupedFloodData["Total_Event_Count"]
groupedFloodData['deathPerEvent_log'] = np.log(groupedFloodData['deathPerEvent'] + 1)  # Adding 1 to avoid log(0) or log(negative)
groupedFloodData['Total_Deaths_Sqrt'] = np.sqrt(groupedFloodData['Total_Deaths'] + 1)  # Adding 1 to avoid log(0) or log(negative)

groupedEarthquakeData = earthquakeAll.groupby(["ISO","Country","Disaster_Subtype"]).agg({"Total_Deaths":"sum", "gii": "mean", "gdi":"mean", "hdi":"mean", "le":"mean", "gnipc":"mean", "Disaster_Group":"count"}).reset_index()
groupedEarthquakeData.columns = ["ISO","Country","Disaster_Subtype","Total_Deaths","gii","gdi","hdi","le","gnipc","Total_Event_Count"]
groupedEarthquakeData["deathPerEvent"] = groupedEarthquakeData["Total_Deaths"] / groupedEarthquakeData["Total_Event_Count"]
groupedEarthquakeData['deathPerEvent_log'] = np.log(groupedEarthquakeData['deathPerEvent'] + 1)  # Adding 1 to avoid log(0) or log(negative)
groupedEarthquakeData['Total_Deaths_Sqrt'] = np.sqrt(groupedEarthquakeData['Total_Deaths'] + 1)  # Adding 1 to avoid log(0) or log(negative)

groupedEarthquakeDataTsunami = earthquakeAll.groupby(["ISO","Country"]).agg({"Total_Deaths":"sum", "gii": "mean", "gdi":"mean", "hdi":"mean", "le":"mean", "gnipc":"mean", "Disaster_Group":"count"}).reset_index()
groupedEarthquakeDataTsunami.columns = ["ISO","Country","Total_Deaths","gii","gdi","hdi","le","gnipc","Total_Event_Count"]
groupedEarthquakeDataTsunami["deathPerEvent"] = groupedEarthquakeDataTsunami["Total_Deaths"] / groupedEarthquakeDataTsunami["Total_Event_Count"]
groupedEarthquakeDataTsunami['deathPerEvent_log'] = np.log(groupedEarthquakeDataTsunami['deathPerEvent'] + 1)  # Adding 1 to avoid log(0) or log(negative)
groupedEarthquakeDataTsunami['Total_Deaths_Sqrt'] = np.sqrt(groupedEarthquakeDataTsunami['Total_Deaths'] + 1)  # Adding 1 to avoid log(0) or log(negative)

groupedStormData = stormDataAll.groupby(["ISO","Country"]).agg({"Total_Deaths":"sum", "gii": "mean", "gdi":"mean","hdi":"mean", "le":"mean","gnipc":"mean","Disaster_Group":"count"}).reset_index()
groupedStormData.columns = ["ISO","Country","Total_Deaths","gii","gdi","hdi","le","gnipc","Total_Event_Count"]
groupedStormData["deathPerEvent"] = groupedStormData["Total_Deaths"] / groupedFloodData["Total_Event_Count"]
groupedStormData['deathPerEvent_log'] = np.log(groupedStormData['deathPerEvent'] + 1)  # Adding 1 to avoid log(0) or log(negative)
groupedStormData['deathPerEventSqrt'] = np.sqrt(groupedStormData['deathPerEvent'] + 1)
groupedStormData['Total_Deaths_Sqrt'] = np.sqrt(groupedStormData['Total_Deaths'] + 1)  # Adding 1 to avoid log(0) or log(negative)

def ytick_formatter(x, pos):
    if x >= 1e6:
        return f'{x / 1e6:.1f}M'
    elif x>= 1e3:
        return f'{x / 1e3:.0f}K'
    else:
        return f'{x:.0f}'
    

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/get-plots')
def get_plots():
    
    ###########################################################################################################################################
    
    fig1 = px.line(decadeDeaths_ByDisaster_Group, x='Decade', y='Total_Deaths', markers=True, title='Total Loss By Decade')
    fig1.update_layout(xaxis_title='Decade', yaxis_title='Total Loss')
    fig1.update_yaxes(ticklabelposition="inside", tickformat=',d')
    fig1.update_layout(width=800, height=600)
    plot1_json = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    
    ###########################################################################################################################################
    
    fig2 = px.bar(decadeDeaths_ByDisaster_Type[decadeDeaths_ByDisaster_Type.Decade >= 1970], x='Decade', y='Total_Deaths', color='Disaster_Type', title='Total Loss By Disaster Type')
    fig2.update_layout(xaxis_title='Year', yaxis_title='Total Loss', legend_title='Category')
    fig2.update_yaxes(ticklabelposition="inside", tickformat=',d')
    fig2.update_layout(width=800, height=600)
    plot2_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    ###########################################################################################################################################

    fig3 = px.bar(yearlyEvents[yearlyEvents.Start_Year >= 1970], x='Start_Year', y='DisNo', title='Total Number of Events By Year')
    fig3.update_layout(xaxis_title='Year', yaxis_title='Total Count')
    fig3.update_yaxes(ticklabelposition="inside", tickformat=',d')
    fig3.update_layout(width=1080, height=600)
    plot3_json = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    ###########################################################################################################################################

    fig4 = px.pie(filteredData, names='Region', title='Proportion Of Region', hole=0.25)
    fig4.update_layout(width=540, height=600)
    plot4_json = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

    ###########################################################################################################################################

    fig5 = px.pie(filteredData, names='Disaster_Type', title='Proportion Of Disaster_Type', hole=0.25)
    fig5.update_layout(width=540, height=600)
    plot5_json = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

    ###########################################################################################################################################

    fig6 = px.choropleth(yearlyRegionalEvents_ByDisasterGroup, locations="ISO",
                     color="DisNo",
                     hover_name="Country",
                     color_continuous_scale=px.colors.sequential.Plasma)
    fig6.update_layout(title_text='Total Regional Events by Disaster Group')
    fig6.update_layout(width=1080, height=900)
    plot6_json = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

    ###########################################################################################################################################

    fig7 = px.choropleth(regionalDeaths_ByDisasterGroup, locations="ISO",
                     color="Total_Deaths",
                     hover_name="Country",
                     color_continuous_scale=px.colors.sequential.Plasma)
    fig7.update_traces(hoverinfo="location+text+z")
    fig7.update_layout(title_text='Total Regional Deaths (log) by Disaster Group')
    fig7.update_layout(width=1080, height=900)
    plot7_json = json.dumps(fig7, cls=plotly.utils.PlotlyJSONEncoder)

    ###########################################################################################################################################
    
    year = 1900
    scl = [[0.0, '#ffffff'],[0.2, '#b4a8ce'],[0.4, '#8573a9'],
        [0.6, '#7159a3'],[0.8, '#5732a1'],[1.0, '#2c0579']]  # purples

    data_slider = []
    for idx, year in enumerate(sorted(yearlyRegionalDeaths_ByDisasterGroup['Start_Year'].unique())):
        df_segmented = yearlyRegionalDeaths_ByDisasterGroup[(yearlyRegionalDeaths_ByDisasterGroup['Start_Year'] == year)]

        for col in df_segmented.columns:
            df_segmented[col] = df_segmented[col].astype(str)

        data_each_yr = dict(
                            type='choropleth',
                            locations=df_segmented['ISO'],
                            z=df_segmented['Total_Deaths'].astype(float),
                            colorscale=scl,
                            colorbar={'title':'# Total Deaths'},
                            name=str(year),
                            legendgroup=str(year),
                            visible=(idx == 0))  # Only the first dataset is visible

        data_slider.append(data_each_yr)

    steps = []
    for i, year in enumerate(sorted(yearlyRegionalDeaths_ByDisasterGroup['Start_Year'].unique())):
        step = dict(
            method='restyle',
            args=['visible', [False] * len(data_slider)],
            label=f'Year {year}'
        )
        step['args'][1][i] = True  # Make only the i-th dataset visible
        steps.append(step)

    sliders = [dict(active=0, pad={"t": 1}, steps=steps)]

    layout = dict(title='Yearly Deaths', 
                geo=dict(scope='world', projection={'type': 'winkel tripel'}),
                sliders=sliders,
                width=1080,
                height=900)

    fig8 = dict(data=data_slider, layout=layout)
    plot8_json = json.dumps(fig8, cls=plotly.utils.PlotlyJSONEncoder)

    ###########################################################################################################################################
    year = 1900
    scl = [[0.0, '#ffffff'],[0.2, '#b4a8ce'],[0.4, '#8573a9'],
        [0.6, '#7159a3'],[0.8, '#5732a1'],[1.0, '#2c0579']] # purples

    data_slider = []
    for idx, year in enumerate(decadelRegionalDeaths_ByDisasterGroup['Decade'].unique()):
        df_segmented =  decadelRegionalDeaths_ByDisasterGroup[(decadelRegionalDeaths_ByDisasterGroup['Decade']== year)]

        for col in df_segmented.columns:
            df_segmented[col] = df_segmented[col].astype(str)

        data_each_yr = dict(
                            type='choropleth',
                            locations = df_segmented['ISO'],
                            z=df_segmented['Total_Deaths'].astype(float),
                            colorscale = scl,
                            colorbar= {'title':'# Total Deaths'},
                            name=str(year),
                            legendgroup=str(year),
                            visible=(idx == 0))  # Only the first dataset is visible

        data_slider.append(data_each_yr)

    steps = []
    for i, year in enumerate(decadelRegionalDeaths_ByDisasterGroup['Decade'].unique()):
        step = dict(
            method='restyle',
            args=['visible', [False] * len(data_slider)],
            label=f'Year {year}'
        )
        step['args'][1][i] = True  # Make only the i-th dataset visible
        steps.append(step)

    sliders = [dict(active=0, pad={"t": 1}, steps=steps)]

    layout = dict(title='Decadel Deaths',
                geo=dict(scope='world', projection={'type': 'winkel tripel'}),
                sliders=sliders,
                width=1080,
                height=900)

    fig9 = dict(data=data_slider, layout=layout)
    plot9_json = json.dumps(fig9, cls=plotly.utils.PlotlyJSONEncoder)

    ###########################################################################################################################################

    sample_df = top_5_rows_per_type.sample(20)

    # Create the table figure
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=list(sample_df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[sample_df[col] for col in sample_df.columns],
                   fill_color='lavender',
                   align='left'))
    ])

    table_fig.update_layout(width=1080, height=720)

    # Convert the figure to JSON
    table1_json = json.dumps(table_fig, cls=plotly.utils.PlotlyJSONEncoder)

    ###########################################################################################################################################

    fig10 = px.scatter(groupedFloodData, x='gii', y='deathPerEvent_log', trendline='ols', 
                 title='Scatter Plot of GII vs DeathPerEventLog with Regression Line', size='Total_Event_Count',
                 hover_name='Country')

    fig10.update_xaxes(title_text='GII')
    fig10.update_yaxes(title_text='Death Per Event (log)')
    fig10.update_traces(name='Regression Line')  # Name the trendline for the legend

    fig10.update_layout(width=1080, height=900)
    plot10_json = json.dumps(fig10, cls=plotly.utils.PlotlyJSONEncoder)
    
    ########################################################################################################################################### 
    
    fig11 = px.histogram(groupedFloodData, x='Total_Event_Count', 
                   title='Histogram of Total Flood Event Count', 
                   labels={'Total_Event_Count': 'Total Event Count', 'count': 'Frequency'})

    fig11.update_layout(width=1080, height=600)
    plot11_json = json.dumps(fig11, cls=plotly.utils.PlotlyJSONEncoder)
    
    ########################################################################################################################################### 
    
    # Creating a 2x3 grid for subplots with 1 scatter plot on top and 3 small histograms below
    fig12 = make_subplots(rows=2, cols=3, 
                        #subplot_titles=('GII vs DeathPerEventLog', '', '', 'Event Count (Lower Third GII)', 'Event Count (Middle Third GII)', 'Event Count (Upper Third GII)'),
                        specs=[[{"colspan": 3, "type": "scatter"}, None, None], [{}, {}, {}]],
                        row_heights=[0.7, 0.3])

    plotData = groupedFloodData
    # Scatter plot (row 1, spans all columns)
    scatter = px.scatter(groupedFloodData, x='gii', y='deathPerEvent_log', size='Total_Event_Count', hover_name='Country', trendline="ols")
    for trace in scatter.data:
        fig12.add_trace(trace, row=1, col=1)

    # Preparing data for histograms
    gii_thirds = groupedFloodData['gii'].quantile([1/3, 2/3]).values

    # Histogram for lower third GII (row 2, col 1)
    fig12.add_trace(go.Histogram(x=groupedFloodData[groupedFloodData['gii'] <= gii_thirds[0]]['Total_Event_Count'], nbinsx=20), row=2, col=1)

    # Histogram for middle third GII (row 2, col 2)
    fig12.add_trace(go.Histogram(x=groupedFloodData[(groupedFloodData['gii'] > gii_thirds[0]) & (groupedFloodData['gii'] <= gii_thirds[1])]['Total_Event_Count'], nbinsx=20), row=2, col=2)

    # Histogram for upper third GII (row 2, col 3)
    fig12.add_trace(go.Histogram(x=groupedFloodData[groupedFloodData['gii'] > gii_thirds[1]]['Total_Event_Count'], nbinsx=20), row=2, col=3)

    # Update layout and axis labels


    fig12.update_yaxes(domain=[0.20, 1], row=1, col=1)
    fig12.update_yaxes(domain=[0.05, 0.15], row=2, col=1)
    fig12.update_yaxes(domain=[0.05, 0.15], row=2, col=2)
    fig12.update_yaxes(domain=[0.05, 0.15], row=2, col=3)
    #fig.update_xaxes(title_text='GII', row=1, col=1)
    fig12.update_yaxes(title_text='Death Per Event (log)', row=1, col=1)
    fig12.update_xaxes(title_text='Event Count (Lower Third GII)', row=2, col=1)
    fig12.update_xaxes(title_text='Event Count (Middle Third GII)', row=2, col=2)
    fig12.update_xaxes(title_text='Event Count (Upper Third GII)', row=2, col=3)
    fig12.update_yaxes(title_text='Frequency', row=2, col=1)
    
    fig12.update_layout(width=1080, height=900, showlegend=False)

    india_data = plotData[plotData['Country'] == 'India'].iloc[0]
    usa_data = plotData[plotData['Country'] == 'United States of America'].iloc[0]
    afghanhistan_data = plotData[plotData['Country'] == 'Afghanistan'].iloc[0]
    china_data = plotData[plotData['Country'] == 'China'].iloc[0]

    # Add annotations with arrows pointing to India and USA
    fig12.add_annotation(
        x=india_data['gii'], y=india_data['deathPerEvent_log'],
        xref="x", yref="y",
        text="India",
        showarrow=True,
        arrowhead=1,
        ax=20,  # Adjust these values to change the arrow's position
        ay=-30, # Adjust these values to change the arrow's position
        row=1, col=1  # Specify the subplot to add this annotation to
    )

    fig12.add_annotation(
        x=usa_data['gii'], y=usa_data['deathPerEvent_log'],
        xref="x", yref="y",
        text="USA",
        showarrow=True,
        arrowhead=1,
        ax=-20, # Adjust these values to change the arrow's position
        ay=-30, # Adjust these values to change the arrow's position
        row=1, col=1  # Specify the subplot to add this annotation to
    )

    fig12.add_annotation(
        x=afghanhistan_data['gii'], y=afghanhistan_data['deathPerEvent_log'],
        xref="x", yref="y",
        text="Afghanistan",
        showarrow=True,
        arrowhead=1,
        ax=-20, # Adjust these values to change the arrow's position
        ay=-30, # Adjust these values to change the arrow's position
        row=1, col=1  # Specify the subplot to add this annotation to
    )

    fig12.add_annotation(
        x=china_data['gii'], y=china_data['deathPerEvent_log'],
        xref="x", yref="y",
        text="China",
        showarrow=True,
        arrowhead=1,
        ax=-20, # Adjust these values to change the arrow's position
        ay=-30, # Adjust these values to change the arrow's position
        row=1, col=1  # Specify the subplot to add this annotation to
    )

    fig12.update_layout(title_text='GII vs Death Per Flood Event (All Countries)')

    plot12_json = json.dumps(fig12, cls=plotly.utils.PlotlyJSONEncoder)
    
    ########################################################################################################################################### 
    
    # Creating a 2x3 grid for subplots with 1 scatter plot on top and 3 small histograms below
    fig13 = make_subplots(rows=2, cols=3, 
                        #subplot_titles=('GII vs DeathPerEventLog', '', '', 'Event Count (Lower Third GII)', 'Event Count (Middle Third GII)', 'Event Count (Upper Third GII)'),
                        specs=[[{"colspan": 3, "type": "scatter"}, None, None], [{}, {}, {}]],
                        row_heights=[0.7, 0.3])

    plotData = groupedFloodData.loc[(groupedFloodData.Total_Event_Count > 20)]
    # Scatter plot (row 1, spans all columns)
    scatter = px.scatter(plotData, x='gii', y='deathPerEvent_log', size='Total_Event_Count', hover_name='Country', trendline="ols")
    for trace in scatter.data:
        fig13.add_trace(trace, row=1, col=1)

    # Preparing data for histograms
    gii_thirds = plotData['gii'].quantile([1/3, 2/3]).values

    # Histogram for lower third GII (row 2, col 1)
    fig13.add_trace(go.Histogram(x=plotData[plotData['gii'] <= gii_thirds[0]]['Total_Event_Count'], nbinsx=20), row=2, col=1)

    # Histogram for middle third GII (row 2, col 2)
    fig13.add_trace(go.Histogram(x=plotData[(plotData['gii'] > gii_thirds[0]) & (plotData['gii'] <= gii_thirds[1])]['Total_Event_Count'], nbinsx=20), row=2, col=2)

    # Histogram for upper third GII (row 2, col 3)
    fig13.add_trace(go.Histogram(x=plotData[plotData['gii'] > gii_thirds[1]]['Total_Event_Count'], nbinsx=20), row=2, col=3)

    # Update layout and axis labels
    fig13.update_layout(width=600, height=900, showlegend=False)


    fig13.update_yaxes(domain=[0.20, 1], row=1, col=1)
    fig13.update_yaxes(domain=[0.05, 0.15], row=2, col=1)
    fig13.update_yaxes(domain=[0.05, 0.15], row=2, col=2)
    fig13.update_yaxes(domain=[0.05, 0.15], row=2, col=3)
    #fig.update_xaxes(title_text='GII', row=1, col=1)
    fig13.update_yaxes(title_text='Death Per Event (log)', row=1, col=1)
    fig13.update_xaxes(title_text='Lower Third GII', row=2, col=1)
    fig13.update_xaxes(title_text='Middle Third GII', row=2, col=2)
    fig13.update_xaxes(title_text='Upper Third GII', row=2, col=3)
    fig13.update_yaxes(title_text='Frequency', row=2, col=1)

    india_data = plotData[plotData['Country'] == 'India'].iloc[0]
    usa_data = plotData[plotData['Country'] == 'United States of America'].iloc[0]
    afghanhistan_data = plotData[plotData['Country'] == 'Afghanistan'].iloc[0]
    china_data = plotData[plotData['Country'] == 'China'].iloc[0]

    # Add annotations with arrows pointing to India and USA
    fig13.add_annotation(
        x=india_data['gii'], y=india_data['deathPerEvent_log'],
        xref="x", yref="y",
        text="India",
        showarrow=True,
        arrowhead=1,
        ax=20,  # Adjust these values to change the arrow's position
        ay=-30, # Adjust these values to change the arrow's position
        row=1, col=1  # Specify the subplot to add this annotation to
    )

    fig13.add_annotation(
        x=usa_data['gii'], y=usa_data['deathPerEvent_log'],
        xref="x", yref="y",
        text="USA",
        showarrow=True,
        arrowhead=1,
        ax=-20, # Adjust these values to change the arrow's position
        ay=-30, # Adjust these values to change the arrow's position
        row=1, col=1  # Specify the subplot to add this annotation to
    )

    fig13.add_annotation(
        x=afghanhistan_data['gii'], y=afghanhistan_data['deathPerEvent_log'],
        xref="x", yref="y",
        text="Afghanistan",
        showarrow=True,
        arrowhead=1,
        ax=-20, # Adjust these values to change the arrow's position
        ay=-30, # Adjust these values to change the arrow's position
        row=1, col=1  # Specify the subplot to add this annotation to
    )

    fig13.add_annotation(
        x=china_data['gii'], y=china_data['deathPerEvent_log'],
        xref="x", yref="y",
        text="China",
        showarrow=True,
        arrowhead=1,
        ax=-20, # Adjust these values to change the arrow's position
        ay=-30, # Adjust these values to change the arrow's position
        row=1, col=1  # Specify the subplot to add this annotation to
    )
    fig13.update_layout(title_text='GII vs Death Per Flood Event (Event# > 20)')

    plot13_json = json.dumps(fig13, cls=plotly.utils.PlotlyJSONEncoder)
    
    ########################################################################################################################################### 
    
    fig14 = make_subplots(rows=2, cols=3, 
                    #subplot_titles=('GII vs DeathPerEventLog', '', '', 'Event Count (Lower Third GII)', 'Event Count (Middle Third GII)', 'Event Count (Upper Third GII)'),
                    specs=[[{"colspan": 3, "type": "scatter"}, None, None], [{}, {}, {}]],
                    row_heights=[0.7, 0.3])

    plotData = groupedFloodData.loc[(groupedFloodData.Total_Event_Count > 20) & (~groupedFloodData.Country.isin(["China","India"]))]
    # Scatter plot (row 1, spans all columns)
    scatter = px.scatter(plotData, x='gii', y='deathPerEvent_log', size='Total_Event_Count', hover_name='Country', trendline="ols")
    for trace in scatter.data:
        fig14.add_trace(trace, row=1, col=1)

    # Preparing data for histograms
    gii_thirds = plotData['gii'].quantile([1/3, 2/3]).values

    # Histogram for lower third GII (row 2, col 1)
    fig14.add_trace(go.Histogram(x=plotData[plotData['gii'] <= gii_thirds[0]]['Total_Event_Count'], nbinsx=20), row=2, col=1)

    # Histogram for middle third GII (row 2, col 2)
    fig14.add_trace(go.Histogram(x=plotData[(plotData['gii'] > gii_thirds[0]) & (plotData['gii'] <= gii_thirds[1])]['Total_Event_Count'], nbinsx=20), row=2, col=2)

    # Histogram for upper third GII (row 2, col 3)
    fig14.add_trace(go.Histogram(x=plotData[plotData['gii'] > gii_thirds[1]]['Total_Event_Count'], nbinsx=20), row=2, col=3)

    # Update layout and axis labels
    fig14.update_layout(width=1080, height=900, showlegend=False)


    fig14.update_yaxes(domain=[0.20, 1], row=1, col=1)
    fig14.update_yaxes(domain=[0.05, 0.15], row=2, col=1)
    fig14.update_yaxes(domain=[0.05, 0.15], row=2, col=2)
    fig14.update_yaxes(domain=[0.05, 0.15], row=2, col=3)
    #fig14.update_xaxes(title_text='GII', row=1, col=1)
    fig14.update_yaxes(title_text='Death Per Event (log)', row=1, col=1)
    fig14.update_xaxes(title_text='Lower Third GII', row=2, col=1)
    fig14.update_xaxes(title_text='Middle Third GII', row=2, col=2)
    fig14.update_xaxes(title_text='Upper Third GII', row=2, col=3)
    fig14.update_yaxes(title_text='Frequency', row=2, col=1)

    countries_data = {
        #"India": plotData[plotData['Country'] == 'India'].iloc[0],
        "USA": plotData[plotData['Country'] == 'United States of America'].iloc[0],
        "Afghanistan": plotData[plotData['Country'] == 'Afghanistan'].iloc[0],
        #"China": plotData[plotData['Country'] == 'China'].iloc[0],
    }

    for country, data in countries_data.items():
        fig14.add_annotation(
            x=data['gii'], y=data['deathPerEvent_log'],
            xref="x", yref="y",
            text=country,
            showarrow=True,
            arrowhead=1,
            ax=20 if country != "USA" else -20,  # Adjust these values based on the country's position
            ay=-30,
            row=1, col=1
        )
    fig13.update_layout(title_text='Not Used')
    plot14_json = json.dumps(fig14, cls=plotly.utils.PlotlyJSONEncoder)
    
    ########################################################################################################################################### 
    fig15 = make_subplots(rows=2, cols=3, 
                    specs=[[{"colspan": 3, "type": "scatter"}, None, None], [{}, {}, {}]],
                    row_heights=[0.7, 0.3])

    plotData = groupedFloodData
    # Scatter plot (row 1, spans all columns) using 'hdi' instead of 'gii'
    scatter = px.scatter(plotData, x='hdi', y='deathPerEvent_log', size='Total_Event_Count', hover_name='Country', trendline="ols")
    for trace in scatter.data:
        fig15.add_trace(trace, row=1, col=1)

    # Preparing data for histograms based on 'hdi'
    hdi_thirds = plotData['hdi'].quantile([1/3, 2/3]).values

    # Histogram for lower third HDI (row 2, col 1)
    fig15.add_trace(go.Histogram(x=plotData[plotData['hdi'] <= hdi_thirds[0]]['Total_Event_Count'], nbinsx=20), row=2, col=1)

    # Histogram for middle third HDI (row 2, col 2)
    fig15.add_trace(go.Histogram(x=plotData[(plotData['hdi'] > hdi_thirds[0]) & (plotData['hdi'] <= hdi_thirds[1])]['Total_Event_Count'], nbinsx=20), row=2, col=2)

    # Histogram for upper third HDI (row 2, col 3)
    fig15.add_trace(go.Histogram(x=plotData[plotData['hdi'] > hdi_thirds[1]]['Total_Event_Count'], nbinsx=20), row=2, col=3)

    # Update layout and axis labels
    fig15.update_layout(width=1080, height=900, showlegend=False)
    fig15.update_yaxes(domain=[0.20, 1], row=1, col=1)
    fig15.update_yaxes(domain=[0.05, 0.15], row=2, col=1)
    fig15.update_yaxes(domain=[0.05, 0.15], row=2, col=2)
    fig15.update_yaxes(domain=[0.05, 0.15], row=2, col=3)
    fig15.update_yaxes(title_text='Death Per Event (log)', row=1, col=1)
    fig15.update_xaxes(title_text='Event Count (Lower Third HDI)', row=2, col=1)
    fig15.update_xaxes(title_text='Event Count (Middle Third HDI)', row=2, col=2)
    fig15.update_xaxes(title_text='Event Count (Upper Third HDI)', row=2, col=3)
    fig15.update_yaxes(title_text='Frequency', row=2, col=1)

    # Annotations for specific countries
    countries_data = {
        "India": plotData[plotData['Country'] == 'India'].iloc[0],
        "USA": plotData[plotData['Country'] == 'United States of America'].iloc[0],
        "Afghanistan": plotData[plotData['Country'] == 'Afghanistan'].iloc[0],
        "China": plotData[plotData['Country'] == 'China'].iloc[0],
    }

    for country, data in countries_data.items():
        fig15.add_annotation(
            x=data['hdi'], y=data['deathPerEvent_log'],
            xref="x", yref="y",
            text=country,
            showarrow=True,
            arrowhead=1,
            ax=20 if country != "USA" else -20,  # Adjust these values based on the country's position
            ay=-30,
            row=1, col=1
        )
    fig15.update_layout(title_text='Not Used')
    plot15_json = json.dumps(fig15, cls=plotly.utils.PlotlyJSONEncoder)

    
    ########################################################################################################################################### 
    
    fig16 = make_subplots(rows=2, cols=3, 
                    specs=[[{"colspan": 3, "type": "scatter"}, None, None], [{}, {}, {}]],
                    row_heights=[0.7, 0.3])

    plotData = groupedFloodData.loc[groupedFloodData.Total_Event_Count > 20]
    # Scatter plot (row 1, spans all columns) using 'hdi' instead of 'gii'
    scatter = px.scatter(plotData, x='hdi', y='deathPerEvent_log', size='Total_Event_Count', hover_name='Country', trendline="ols")
    for trace in scatter.data:
        fig16.add_trace(trace, row=1, col=1)

    # Preparing data for histograms based on 'hdi'
    hdi_thirds = plotData['hdi'].quantile([1/3, 2/3]).values

    # Histogram for lower third HDI (row 2, col 1)
    fig16.add_trace(go.Histogram(x=plotData[plotData['hdi'] <= hdi_thirds[0]]['Total_Event_Count'], nbinsx=20), row=2, col=1)

    # Histogram for middle third HDI (row 2, col 2)
    fig16.add_trace(go.Histogram(x=plotData[(plotData['hdi'] > hdi_thirds[0]) & (plotData['hdi'] <= hdi_thirds[1])]['Total_Event_Count'], nbinsx=20), row=2, col=2)

    # Histogram for upper third HDI (row 2, col 3)
    fig16.add_trace(go.Histogram(x=plotData[plotData['hdi'] > hdi_thirds[1]]['Total_Event_Count'], nbinsx=20), row=2, col=3)

    # Update layout and axis labels
    fig16.update_layout(width=600, height=900, showlegend=False)
    fig16.update_yaxes(domain=[0.20, 1], row=1, col=1)
    fig16.update_yaxes(domain=[0.05, 0.15], row=2, col=1)
    fig16.update_yaxes(domain=[0.05, 0.15], row=2, col=2)
    fig16.update_yaxes(domain=[0.05, 0.15], row=2, col=3)
    fig16.update_yaxes(title_text='Death Per Event (log)', row=1, col=1)
    fig16.update_xaxes(title_text='Lower Third HDI', row=2, col=1)
    fig16.update_xaxes(title_text='Middle Third HDI', row=2, col=2)
    fig16.update_xaxes(title_text='Upper Third HDI', row=2, col=3)
    fig16.update_yaxes(title_text='Frequency', row=2, col=1)

    # Annotations for specific countries
    countries_data = {
        "India": plotData[plotData['Country'] == 'India'].iloc[0],
        "USA": plotData[plotData['Country'] == 'United States of America'].iloc[0],
        "Afghanistan": plotData[plotData['Country'] == 'Afghanistan'].iloc[0],
        "China": plotData[plotData['Country'] == 'China'].iloc[0],
    }

    for country, data in countries_data.items():
        fig16.add_annotation(
            x=data['hdi'], y=data['deathPerEvent_log'],
            xref="x", yref="y",
            text=country,
            showarrow=True,
            arrowhead=1,
            ax=20 if country != "USA" else -20,  # Adjust these values based on the country's position
            ay=-30,
            row=1, col=1
        )
    fig16.update_layout(title_text='HDI vs Death Per Flood Event (Event# > 20)')
    plot16_json = json.dumps(fig16, cls=plotly.utils.PlotlyJSONEncoder)

    ########################################################################################################################################### 

    fig17 = make_subplots(rows=2, cols=3, 
                    specs=[[{"colspan": 3, "type": "scatter"}, None, None], [{}, {}, {}]],
                    row_heights=[0.7, 0.3])

    plotData = groupedFloodData.loc[(groupedFloodData.Total_Event_Count > 20)]
    # Scatter plot (row 1, spans all columns) using 'gnipc'
    scatter = px.scatter(plotData, x='gnipc', y='deathPerEvent_log', size='Total_Event_Count', hover_name='Country', trendline="ols")
    for trace in scatter.data:
        fig17.add_trace(trace, row=1, col=1)

    # Preparing data for histograms based on 'gnipc'
    gnipc_thirds = plotData['gnipc'].quantile([1/3, 2/3]).values

    # Histogram for lower third GNIPC (row 2, col 1)
    fig17.add_trace(go.Histogram(x=plotData[plotData['gnipc'] <= gnipc_thirds[0]]['Total_Event_Count'], nbinsx=20), row=2, col=1)

    # Histogram for middle third GNIPC (row 2, col 2)
    fig17.add_trace(go.Histogram(x=plotData[(plotData['gnipc'] > gnipc_thirds[0]) & (plotData['gnipc'] <= gnipc_thirds[1])]['Total_Event_Count'], nbinsx=20), row=2, col=2)

    # Histogram for upper third GNIPC (row 2, col 3)
    fig17.add_trace(go.Histogram(x=plotData[plotData['gnipc'] > gnipc_thirds[1]]['Total_Event_Count'], nbinsx=20), row=2, col=3)

    # Update layout and axis labels
    fig17.update_layout(width=600, height=900, showlegend=False)
    fig17.update_yaxes(domain=[0.20, 1], row=1, col=1)
    fig17.update_yaxes(domain=[0.05, 0.15], row=2, col=1)
    fig17.update_yaxes(domain=[0.05, 0.15], row=2, col=2)
    fig17.update_yaxes(domain=[0.05, 0.15], row=2, col=3)
    fig17.update_yaxes(title_text='Death Per Event (log)', row=1, col=1)
    fig17.update_xaxes(title_text='Lower Third GNIPC', row=2, col=1)
    fig17.update_xaxes(title_text='Middle Third GNIPC', row=2, col=2)
    fig17.update_xaxes(title_text='Upper Third GNIPC', row=2, col=3)
    fig17.update_yaxes(title_text='Frequency', row=2, col=1)

    countries_data = {
        "India": plotData[plotData['Country'] == 'India'].iloc[0],
        "USA": plotData[plotData['Country'] == 'United States of America'].iloc[0],
        "Afghanistan": plotData[plotData['Country'] == 'Afghanistan'].iloc[0],
        "China": plotData[plotData['Country'] == 'China'].iloc[0],
    }
    # Annotations for specific countries using 'gnipc'
    for country, data in countries_data.items():
        fig17.add_annotation(
            x=data['gnipc'], y=data['deathPerEvent_log'],
            xref="x", yref="y",
            text=country,
            showarrow=True,
            arrowhead=1,
            ax=20 if country != "Phillippines" else 90,  # Adjust these values based on the country's position
            ay=-30,
            row=1, col=1
        )
    fig17.update_layout(title_text='GNIPC vs Death Per Flood Event (Event# > 20)')
    plot17_json = json.dumps(fig17, cls=plotly.utils.PlotlyJSONEncoder)

    ########################################################################################################################################### 
    
    fig18 = px.histogram(groupedEarthquakeDataTsunami, x='Total_Event_Count', 
                   title='Histogram of Total Earthquake Event Count', 
                   labels={'Total_Event_Count': 'Total Event Count', 'count': 'Frequency'})

    fig18.update_layout(width=1080, height=600)

    plot18_json = json.dumps(fig18, cls=plotly.utils.PlotlyJSONEncoder)
        
    ########################################################################################################################################### 
    
    fig19 = make_subplots(rows=2, cols=3, 
                    specs=[[{"colspan": 3, "type": "scatter"}, None, None], [{}, {}, {}]],
                    row_heights=[0.7, 0.3])
    plotData = groupedEarthquakeDataTsunami.loc[(groupedEarthquakeDataTsunami.Total_Event_Count > 10)] 

    # Scatter plot (row 1, spans all columns) using 'hdi' instead of 'gii'
    scatter = px.scatter(plotData, x='hdi', y='deathPerEvent_log', size='Total_Event_Count', hover_name='Country', trendline="ols")
    for trace in scatter.data:
        fig19.add_trace(trace, row=1, col=1)

    # Preparing data for histograms based on 'hdi'
    hdi_thirds = plotData['hdi'].quantile([1/3, 2/3]).values

    # Histogram for lower third HDI (row 2, col 1)
    fig19.add_trace(go.Histogram(x=plotData[plotData['hdi'] <= hdi_thirds[0]]['Total_Event_Count'], nbinsx=20), row=2, col=1)

    # Histogram for middle third HDI (row 2, col 2)
    fig19.add_trace(go.Histogram(x=plotData[(plotData['hdi'] > hdi_thirds[0]) & (plotData['hdi'] <= hdi_thirds[1])]['Total_Event_Count'], nbinsx=20), row=2, col=2)

    # Histogram for upper third HDI (row 2, col 3)
    fig19.add_trace(go.Histogram(x=plotData[plotData['hdi'] > hdi_thirds[1]]['Total_Event_Count'], nbinsx=20), row=2, col=3)

    # Update layout and axis labels
    fig19.update_layout(width=1080, height=900, showlegend=False)
    fig19.update_yaxes(domain=[0.20, 1], row=1, col=1)
    fig19.update_yaxes(domain=[0.05, 0.15], row=2, col=1)
    fig19.update_yaxes(domain=[0.05, 0.15], row=2, col=2)
    fig19.update_yaxes(domain=[0.05, 0.15], row=2, col=3)
    fig19.update_yaxes(title_text='Death Per Event (log)', row=1, col=1)
    fig19.update_xaxes(title_text='Lower Third HDI', row=2, col=1)
    fig19.update_xaxes(title_text='Middle Third HDI', row=2, col=2)
    fig19.update_xaxes(title_text='Upper Third HDI', row=2, col=3)
    fig19.update_yaxes(title_text='Frequency', row=2, col=1)

    # Annotations for specific countries
    countries_data = {
        "Indonesia": plotData[plotData['Country'] == 'Indonesia'].iloc[0],
        "USA": plotData[plotData['Country'] == 'United States of America'].iloc[0],
        "Japan": plotData[plotData['Country'] == 'Japan'].iloc[0],
        "China": plotData[plotData['Country'] == 'China'].iloc[0],
        "Türkiye": plotData[plotData['Country'] == 'Türkiye'].iloc[0],
    }

    for country, data in countries_data.items():
        fig19.add_annotation(
            x=data['hdi'], y=data['deathPerEvent_log'],
            xref="x", yref="y",
            text=country,
            showarrow=True,
            arrowhead=1,
            ax=20 if country != "USA" else -20,  # Adjust these values based on the country's position
            ay=-30,
            row=1, col=1
        )

    fig19.update_layout(title_text='HDI vs Death Per Earthquake Event Tsunami Included (Event# > 10)')
    plot19_json = json.dumps(fig19, cls=plotly.utils.PlotlyJSONEncoder)
    
    ########################################################################################################################################### 
    
    # Creating a 2x3 grid for subplots with 1 scatter plot on top and 3 small histograms below
    fig20 = make_subplots(rows=2, cols=3, 
                        specs=[[{"colspan": 3, "type": "scatter"}, None, None], [{}, {}, {}]],
                        row_heights=[0.7, 0.3])
    plotData = groupedEarthquakeData.loc[(groupedEarthquakeData.Total_Event_Count > 10) & (groupedEarthquakeData.Disaster_Subtype == "Ground movement")] 

    # Scatter plot (row 1, spans all columns) using 'hdi' instead of 'gii'
    scatter = px.scatter(plotData, x='hdi', y='deathPerEvent_log', size='Total_Event_Count', hover_name='Country', trendline="ols")
    for trace in scatter.data:
        fig20.add_trace(trace, row=1, col=1)

    # Preparing data for histograms based on 'hdi'
    hdi_thirds = plotData['hdi'].quantile([1/3, 2/3]).values

    # Histogram for lower third HDI (row 2, col 1)
    fig20.add_trace(go.Histogram(x=plotData[plotData['hdi'] <= hdi_thirds[0]]['Total_Event_Count'], nbinsx=20), row=2, col=1)

    # Histogram for middle third HDI (row 2, col 2)
    fig20.add_trace(go.Histogram(x=plotData[(plotData['hdi'] > hdi_thirds[0]) & (plotData['hdi'] <= hdi_thirds[1])]['Total_Event_Count'], nbinsx=20), row=2, col=2)

    # Histogram for upper third HDI (row 2, col 3)
    fig20.add_trace(go.Histogram(x=plotData[plotData['hdi'] > hdi_thirds[1]]['Total_Event_Count'], nbinsx=20), row=2, col=3)

    # Update layout and axis labels
    fig20.update_layout(width=600, height=900, showlegend=False)
    fig20.update_yaxes(domain=[0.20, 1], row=1, col=1)
    fig20.update_yaxes(domain=[0.05, 0.15], row=2, col=1)
    fig20.update_yaxes(domain=[0.05, 0.15], row=2, col=2)
    fig20.update_yaxes(domain=[0.05, 0.15], row=2, col=3)
    fig20.update_yaxes(title_text='Death Per Event (log)', row=1, col=1)
    fig20.update_xaxes(title_text='Lower Third HDI', row=2, col=1)
    fig20.update_xaxes(title_text='Middle Third HDI', row=2, col=2)
    fig20.update_xaxes(title_text='Upper Third HDI', row=2, col=3)
    fig20.update_yaxes(title_text='Frequency', row=2, col=1)

    # Annotations for specific countries
    countries_data = {
        "Indonesia": plotData[plotData['Country'] == 'Indonesia'].iloc[0],
        "USA": plotData[plotData['Country'] == 'United States of America'].iloc[0],
        "Japan": plotData[plotData['Country'] == 'Japan'].iloc[0],
        "China": plotData[plotData['Country'] == 'China'].iloc[0],
        "Türkiye": plotData[plotData['Country'] == 'Türkiye'].iloc[0],
    }

    for country, data in countries_data.items():
        fig20.add_annotation(
            x=data['hdi'], y=data['deathPerEvent_log'],
            xref="x", yref="y",
            text=country,
            showarrow=True,
            arrowhead=1,
            ax=20 if country != "USA" else -20,  # Adjust these values based on the country's position
            ay=-30,
            row=1, col=1
        )
    fig20.update_layout(title_text='HDI vs Death Per Earthquake Event Only Ground Movement (Event# > 10)')
    plot20_json = json.dumps(fig20, cls=plotly.utils.PlotlyJSONEncoder)

    
    ########################################################################################################################################### 
    fig21 = make_subplots(rows=2, cols=3, 
                    specs=[[{"colspan": 3, "type": "scatter"}, None, None], [{}, {}, {}]],
                    row_heights=[0.7, 0.3])

    plotData = groupedEarthquakeData.loc[(groupedEarthquakeData.Total_Event_Count > 10) & (groupedEarthquakeData.Disaster_Subtype == "Ground movement")]
    # Scatter plot (row 1, spans all columns) using 'gii'
    scatter = px.scatter(plotData, x='gii', y='deathPerEvent_log', size='Total_Event_Count', hover_name='Country', trendline="ols")
    for trace in scatter.data:
        fig21.add_trace(trace, row=1, col=1)

    # Preparing data for histograms based on 'gii'
    gii_thirds = plotData['gii'].quantile([1/3, 2/3]).values

    # Histogram for lower third GII (row 2, col 1)
    fig21.add_trace(go.Histogram(x=plotData[plotData['gii'] <= gii_thirds[0]]['Total_Event_Count'], nbinsx=20), row=2, col=1)

    # Histogram for middle third GII (row 2, col 2)
    fig21.add_trace(go.Histogram(x=plotData[(plotData['gii'] > gii_thirds[0]) & (plotData['gii'] <= gii_thirds[1])]['Total_Event_Count'], nbinsx=20), row=2, col=2)

    # Histogram for upper third GII (row 2, col 3)
    fig21.add_trace(go.Histogram(x=plotData[plotData['gii'] > gii_thirds[1]]['Total_Event_Count'], nbinsx=20), row=2, col=3)

    # Update layout and axis labels
    fig21.update_layout(width=600, height=900, showlegend=False)
    fig21.update_yaxes(domain=[0.20, 1], row=1, col=1)
    fig21.update_yaxes(domain=[0.05, 0.15], row=2, col=1)
    fig21.update_yaxes(domain=[0.05, 0.15], row=2, col=2)
    fig21.update_yaxes(domain=[0.05, 0.15], row=2, col=3)
    fig21.update_yaxes(title_text='Death Per Event (log)', row=1, col=1)
    fig21.update_xaxes(title_text='Lower Third GII', row=2, col=1)
    fig21.update_xaxes(title_text='Middle Third GII', row=2, col=2)
    fig21.update_xaxes(title_text='Upper Third GII', row=2, col=3)
    fig21.update_yaxes(title_text='Frequency', row=2, col=1)

    countries_data = {
        "Indonesia": plotData[plotData['Country'] == 'Indonesia'].iloc[0],
        "USA": plotData[plotData['Country'] == 'United States of America'].iloc[0],
        "Japan": plotData[plotData['Country'] == 'Japan'].iloc[0],
        "China": plotData[plotData['Country'] == 'China'].iloc[0],
        "Türkiye": plotData[plotData['Country'] == 'Türkiye'].iloc[0],
    }

    # Annotations for specific countries using 'gii' values
    for country, data in countries_data.items():
        fig21.add_annotation(
            x=data['gii'], y=data['deathPerEvent_log'],
            xref="x", yref="y",
            text=country,
            showarrow=True,
            arrowhead=1,
            ax=20 if country != "USA" else -20,  # Adjust these values based on the country's position
            ay=-30,
            row=1, col=1
        )
    fig21.update_layout(title_text='GII vs Death Per Earthquake Event Only Ground Movement (Event# > 10)')
    plot21_json = json.dumps(fig21, cls=plotly.utils.PlotlyJSONEncoder)

    ########################################################################################################################################### 

    fig22 = make_subplots(rows=2, cols=3, 
                    specs=[[{"colspan": 3, "type": "scatter"}, None, None], [{}, {}, {}]],
                    row_heights=[0.7, 0.3])

    plotData = groupedEarthquakeData.loc[(groupedEarthquakeData.Total_Event_Count > 10) & (groupedEarthquakeData.Disaster_Subtype == "Ground movement")]
    # Scatter plot (row 1, spans all columns) using 'gnipc'
    scatter = px.scatter(plotData, x='gnipc', y='deathPerEvent_log', size='Total_Event_Count', hover_name='Country', trendline="ols")
    for trace in scatter.data:
        fig22.add_trace(trace, row=1, col=1)

    # Preparing data for histograms based on 'gnipc'
    gnipc_thirds = plotData['gnipc'].quantile([1/3, 2/3]).values

    # Histogram for lower third GNIPC (row 2, col 1)
    fig22.add_trace(go.Histogram(x=plotData[plotData['gnipc'] <= gnipc_thirds[0]]['Total_Event_Count'], nbinsx=20), row=2, col=1)

    # Histogram for middle third GNIPC (row 2, col 2)
    fig22.add_trace(go.Histogram(x=plotData[(plotData['gnipc'] > gnipc_thirds[0]) & (plotData['gnipc'] <= gnipc_thirds[1])]['Total_Event_Count'], nbinsx=20), row=2, col=2)

    # Histogram for upper third GNIPC (row 2, col 3)
    fig22.add_trace(go.Histogram(x=plotData[plotData['gnipc'] > gnipc_thirds[1]]['Total_Event_Count'], nbinsx=20), row=2, col=3)

    # Update layout and axis labels
    fig22.update_layout(width=600, height=900, showlegend=False)
    fig22.update_yaxes(domain=[0.20, 1], row=1, col=1)
    fig22.update_yaxes(domain=[0.05, 0.15], row=2, col=1)
    fig22.update_yaxes(domain=[0.05, 0.15], row=2, col=2)
    fig22.update_yaxes(domain=[0.05, 0.15], row=2, col=3)
    fig22.update_yaxes(title_text='Death Per Event (log)', row=1, col=1)
    fig22.update_xaxes(title_text='Lower Third GNIPC', row=2, col=1)
    fig22.update_xaxes(title_text='Middle Third GNIPC', row=2, col=2)
    fig22.update_xaxes(title_text='Upper Third GNIPC', row=2, col=3)
    fig22.update_yaxes(title_text='Frequency', row=2, col=1)

    countries_data = {
        "Indonesia": plotData[plotData['Country'] == 'Indonesia'].iloc[0],
        "USA": plotData[plotData['Country'] == 'United States of America'].iloc[0],
        "Japan": plotData[plotData['Country'] == 'Japan'].iloc[0],
        "China": plotData[plotData['Country'] == 'China'].iloc[0],
        "Türkiye": plotData[plotData['Country'] == 'Türkiye'].iloc[0],
    }

    # Annotations for specific countries using 'gnipc'
    for country, data in countries_data.items():
        fig22.add_annotation(
            x=data['gnipc'], y=data['deathPerEvent_log'],
            xref="x", yref="y",
            text=country,
            showarrow=True,
            arrowhead=1,
            ax=20 if country != "Phillippines" else 90,  # Adjust these values based on the country's position
            ay=-30,
            row=1, col=1
        )
    fig22.update_layout(title_text='GNIPC vs Death Per Earthquake Event Only Ground Movement (Event# > 10)')
    plot22_json = json.dumps(fig22, cls=plotly.utils.PlotlyJSONEncoder)

    
    ########################################################################################################################################### 
    
    fig23 = px.histogram(groupedStormData, x='Total_Event_Count', 
                   title='Histogram of Total Storm Event Count', 
                   labels={'Total_Event_Count': 'Total Event Count', 'count': 'Frequency'})

    fig23.update_layout(width=1080, height=600)
    plot23_json = json.dumps(fig23, cls=plotly.utils.PlotlyJSONEncoder)

    ########################################################################################################################################### 
    
    # Creating a 2x3 grid for subplots with 1 scatter plot on top and 3 small histograms below
    fig24 = make_subplots(rows=2, cols=3, 
                        specs=[[{"colspan": 3, "type": "scatter"}, None, None], [{}, {}, {}]],
                        row_heights=[0.7, 0.3])

    plotData = groupedStormData.loc[groupedStormData.Total_Event_Count >= 20]
    # Scatter plot (row 1, spans all columns) using 'hdi' instead of 'gii'
    scatter = px.scatter(plotData, x='hdi', y='deathPerEvent_log', size='Total_Event_Count', hover_name='Country', trendline="ols")
    for trace in scatter.data:
        fig24.add_trace(trace, row=1, col=1)

    # Preparing data for histograms based on 'hdi'
    hdi_thirds = plotData['hdi'].quantile([1/3, 2/3]).values

    # Histogram for lower third HDI (row 2, col 1)
    fig24.add_trace(go.Histogram(x=plotData[plotData['hdi'] <= hdi_thirds[0]]['Total_Event_Count'], nbinsx=20), row=2, col=1)

    # Histogram for middle third HDI (row 2, col 2)
    fig24.add_trace(go.Histogram(x=plotData[(plotData['hdi'] > hdi_thirds[0]) & (plotData['hdi'] <= hdi_thirds[1])]['Total_Event_Count'], nbinsx=20), row=2, col=2)

    # Histogram for upper third HDI (row 2, col 3)
    fig24.add_trace(go.Histogram(x=plotData[plotData['hdi'] > hdi_thirds[1]]['Total_Event_Count'], nbinsx=20), row=2, col=3)

    # Update layout and axis labels
    fig24.update_layout(width=600, height=900, showlegend=False)
    fig24.update_yaxes(domain=[0.20, 1], row=1, col=1)
    fig24.update_yaxes(domain=[0.05, 0.15], row=2, col=1)
    fig24.update_yaxes(domain=[0.05, 0.15], row=2, col=2)
    fig24.update_yaxes(domain=[0.05, 0.15], row=2, col=3)
    fig24.update_yaxes(title_text='Death Per Event (log)', row=1, col=1)
    fig24.update_xaxes(title_text='Lower Third HDI', row=2, col=1)
    fig24.update_xaxes(title_text='Middle Third HDI', row=2, col=2)
    fig24.update_xaxes(title_text='Upper Third HDI', row=2, col=3)
    fig24.update_yaxes(title_text='Frequency', row=2, col=1)

    # Annotations for specific countries
    countries_data = {
        #"Indonesia": plotData[plotData['Country'] == 'Indonesia'].iloc[0],
        "USA": plotData[plotData['Country'] == 'United States of America'].iloc[0],
        "Phillippines": plotData[plotData['Country'] == 'Philippines'].iloc[0],
        "China": plotData[plotData['Country'] == 'China'].iloc[0],
        #"Türkiye": plotData[plotData['Country'] == 'Türkiye'].iloc[0],
    }

    for country, data in countries_data.items():
        fig24.add_annotation(
            x=data['hdi'], y=data['deathPerEvent_log'],
            xref="x", yref="y",
            text=country,
            showarrow=True,
            arrowhead=1,
            ax=20 if country != "Phillippines" else 90,  # Adjust these values based on the country's position
            ay=-30,
            row=1, col=1
        )
        
    fig24.update_layout(title_text='HDI vs Death Per Storm Event (Event# > 20)')
    plot24_json = json.dumps(fig24, cls=plotly.utils.PlotlyJSONEncoder)


    ########################################################################################################################################### 
    
    fig25 = make_subplots(rows=2, cols=3, 
                    specs=[[{"colspan": 3, "type": "scatter"}, None, None], [{}, {}, {}]],
                    row_heights=[0.7, 0.3])

    plotData = groupedStormData.loc[groupedStormData.Total_Event_Count >= 20]
    # Scatter plot (row 1, spans all columns) using 'gii'
    scatter = px.scatter(plotData, x='gii', y='deathPerEvent_log', size='Total_Event_Count', hover_name='Country', trendline="ols")
    for trace in scatter.data:
        fig25.add_trace(trace, row=1, col=1)

    # Preparing data for histograms based on 'gii'
    gii_thirds = plotData['gii'].quantile([1/3, 2/3]).values

    # Histogram for lower third GII (row 2, col 1)
    fig25.add_trace(go.Histogram(x=plotData[plotData['gii'] <= gii_thirds[0]]['Total_Event_Count'], nbinsx=20), row=2, col=1)

    # Histogram for middle third GII (row 2, col 2)
    fig25.add_trace(go.Histogram(x=plotData[(plotData['gii'] > gii_thirds[0]) & (plotData['gii'] <= gii_thirds[1])]['Total_Event_Count'], nbinsx=20), row=2, col=2)

    # Histogram for upper third GII (row 2, col 3)
    fig25.add_trace(go.Histogram(x=plotData[plotData['gii'] > gii_thirds[1]]['Total_Event_Count'], nbinsx=20), row=2, col=3)

    # Update layout and axis labels
    fig25.update_layout(width=600, height=900, showlegend=False)
    fig25.update_yaxes(domain=[0.20, 1], row=1, col=1)
    fig25.update_yaxes(domain=[0.05, 0.15], row=2, col=1)
    fig25.update_yaxes(domain=[0.05, 0.15], row=2, col=2)
    fig25.update_yaxes(domain=[0.05, 0.15], row=2, col=3)
    fig25.update_yaxes(title_text='Death Per Event (log)', row=1, col=1)
    fig25.update_xaxes(title_text='Lower Third GII', row=2, col=1)
    fig25.update_xaxes(title_text='Middle Third GII', row=2, col=2)
    fig25.update_xaxes(title_text='Upper Third GII', row=2, col=3)
    fig25.update_yaxes(title_text='Frequency', row=2, col=1)

    countries_data = {
        #"Indonesia": plotData[plotData['Country'] == 'Indonesia'].iloc[0],
        "USA": plotData[plotData['Country'] == 'United States of America'].iloc[0],
        "Phillippines": plotData[plotData['Country'] == 'Philippines'].iloc[0],
        "China": plotData[plotData['Country'] == 'China'].iloc[0],
        #"Türkiye": plotData[plotData['Country'] == 'Türkiye'].iloc[0],
    }

    # Annotations for specific countries using 'gii'
    for country, data in countries_data.items():
        fig25.add_annotation(
            x=data['gii'], y=data['deathPerEvent_log'],
            xref="x", yref="y",
            text=country,
            showarrow=True,
            arrowhead=1,
            ax=20 if country != "Phillippines" else 90,  # Adjust these values based on the country's position
            ay=-30,
            row=1, col=1
        )

    fig25.update_layout(title_text='GII vs Death Per Storm Event (Event# > 20)')
    plot25_json = json.dumps(fig25, cls=plotly.utils.PlotlyJSONEncoder)

    
    ########################################################################################################################################### 
    
    fig26 = make_subplots(rows=2, cols=3, 
                    specs=[[{"colspan": 3, "type": "scatter"}, None, None], [{}, {}, {}]],
                    row_heights=[0.7, 0.3])

    plotData = groupedStormData.loc[groupedStormData.Total_Event_Count >= 20]
    # Scatter plot (row 1, spans all columns) using 'gnipc'
    scatter = px.scatter(plotData, x='gnipc', y='deathPerEvent_log', size='Total_Event_Count', hover_name='Country', trendline="ols")
    for trace in scatter.data:
        fig26.add_trace(trace, row=1, col=1)

    # Preparing data for histograms based on 'gnipc'
    gnipc_thirds = plotData['gnipc'].quantile([1/3, 2/3]).values

    # Histogram for lower third GNIPC (row 2, col 1)
    fig26.add_trace(go.Histogram(x=plotData[plotData['gnipc'] <= gnipc_thirds[0]]['Total_Event_Count'], nbinsx=20), row=2, col=1)

    # Histogram for middle third GNIPC (row 2, col 2)
    fig26.add_trace(go.Histogram(x=plotData[(plotData['gnipc'] > gnipc_thirds[0]) & (plotData['gnipc'] <= gnipc_thirds[1])]['Total_Event_Count'], nbinsx=20), row=2, col=2)

    # Histogram for upper third GNIPC (row 2, col 3)
    fig26.add_trace(go.Histogram(x=plotData[plotData['gnipc'] > gnipc_thirds[1]]['Total_Event_Count'], nbinsx=20), row=2, col=3)

    # Update layout and axis labels
    fig26.update_layout(width=600, height=900, showlegend=False)
    fig26.update_yaxes(domain=[0.20, 1], row=1, col=1)
    fig26.update_yaxes(domain=[0.05, 0.15], row=2, col=1)
    fig26.update_yaxes(domain=[0.05, 0.15], row=2, col=2)
    fig26.update_yaxes(domain=[0.05, 0.15], row=2, col=3)
    fig26.update_yaxes(title_text='Death Per Event (log)', row=1, col=1)
    fig26.update_xaxes(title_text='Lower Third GNIPC', row=2, col=1)
    fig26.update_xaxes(title_text='Middle Third GNIPC', row=2, col=2)
    fig26.update_xaxes(title_text='Upper Third GNIPC', row=2, col=3)
    fig26.update_yaxes(title_text='Frequency', row=2, col=1)

    countries_data = {
        #"Indonesia": plotData[plotData['Country'] == 'Indonesia'].iloc[0],
        "USA": plotData[plotData['Country'] == 'United States of America'].iloc[0],
        "Phillippines": plotData[plotData['Country'] == 'Philippines'].iloc[0],
        "China": plotData[plotData['Country'] == 'China'].iloc[0],
        #"Türkiye": plotData[plotData['Country'] == 'Türkiye'].iloc[0],
    }

    # Annotations for specific countries using 'gnipc'
    for country, data in countries_data.items():
        fig26.add_annotation(
            x=data['gnipc'], y=data['deathPerEvent_log'],
            xref="x", yref="y",
            text=country,
            showarrow=True,
            arrowhead=1,
            ax=20 if country != "Phillippines" else 90,  # Adjust these values based on the country's position
            ay=-30,
            row=1, col=1
        )
    fig26.update_layout(title_text='GNIPC vs Death Per Storm Event (Event# > 20)')
    plot26_json = json.dumps(fig26, cls=plotly.utils.PlotlyJSONEncoder)
    
    ########################################################################################################################################### 
    
    
    
    explanations = [
        "Explanation for Plot 1: This plot shows...",
        "Explanation for Plot 2: This plot illustrates...",
        # Add explanations for all plots and the table
        "Explanation for the Table: The table displays...",
    ]

    # Combine plots and explanations in a list of tuples
    plots_and_explanations = [
                            {"plot": json.loads(plot1_json), "explanation": explanations[0]},
                            {"plot": json.loads(plot2_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot3_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot4_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot5_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot6_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot7_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot8_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot9_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot10_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot11_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot12_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot13_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot14_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot15_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot16_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot17_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot18_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot19_json), "explanation": explanations[-1]},
                            {"plot": json.loads(plot20_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot21_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot22_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot23_json), "explanation": explanations[-1]},
                            {"plot": json.loads(plot24_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot25_json), "explanation": explanations[1]},
                            {"plot": json.loads(plot26_json), "explanation": explanations[1]}
                            ]

    return jsonify(plots_and_explanations)


if __name__ == '__main__':
    app.run(debug=False)
