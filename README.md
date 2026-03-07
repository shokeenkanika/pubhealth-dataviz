# pubhealth-dataviz
This repository showcases some exploratory, descriptive plots showcasing trends in drug abuse by geography, race, age, urbanization and sex across US counties using Multiple Causes of Death data from the Center for Disease Control (CDC). It was originally created in completion of final project requirements for PPHA 30538 Data and Visualization.

# Data
The data files necessary to run the code scripts in this project can be found at the following Box link:- 
https://uchicago.box.com/s/1kebzzyeu25or4dqfza7vzu1n7qbbx4e 

To ensure that all files run as expected, please replace this folder as downloaded into the 'data' folder in your local clone of our repository. Do not change any folders. For reference, the expected folder structure should look like

yourlocalreponame / data / derived / geo
                  / outputs / figures / altair_figures
                                      / baseline_death_maps
                                      / heterogeneity_maps
                                      / policy_maps
                  / raw / dataverse_files
                        / mcd-cdc
                        / spatial-county

The authors give due credit to the open-source platforms that enabled the acquisition of our raw data and made this project possible as listed below - 

## 1. CDC Multiple Cause of Death (MCOD) Data 
https://wonder.cdc.gov/mcd-icd10-expanded.html 

We requested MCOD data from the CDC Wonder website using three underlying death codes X40-X44 (Unintentional poisoning), X60-X64 (Suicide poisoning) and Y10-Y14 (Undetermined intent). We chose to drop X85 (Homicide/Assalt poisoning), though it is bundled with the other death codes listed in other common literature on this subject, but we did not wish to include crime-related mortality. Multiple pulls were undertaken, separated by drug type, age, sex, urbanization, and race. We filtered by a type of the aforementioned categories, named the file as such, and then continued the pull for the next type. For example, the first pull was County x Year x Sex, filtered by 2018 and T40.1. All pulls are being aggregated in preprocessing.py.

The pulls had to be so detailed because the CDC Wonder portal limits each pull to 75,000 rows. Thus, any replication attempt must follow these instructions exactly. 

The underlying death codes of relevance were: 
1. T40.1: Heroine
2. T40.2: Natsemi
3. T40.3: Methadone
4. T40.4: Synthopiods (Synthetic Opiods) 
5. T40.5: Cocaine
6. T40.36: Psychostim (Pyschostimulants) 

## 2. State Cannabis Policies Bundle 
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2SB7ZF 

This dataset contains information on cannabis policy, categorizing all 50 U.S. States by type of policy on cannabis from 1994 to 2023. It contains measure for three cannabis policy bundles: pharmaceurical, permissive, and fiscal. We used this data to create our policy csv file called 'policy_state_2018_2023_minimal.csv' in which we have used the information found in this dataset to categorize all 50 states into one of three policy groups - All adult use allowed, only medical-use allowed and no adult/medical use allowed. The dataset is duly cited as follows: 

Mallinson, Daniel; Richardson, Lilliard E. Jr.; Neeley, Grant W.; Altaf, Shazib, 2024, "State Cannabis Policy Bundles", https://doi.org/10.7910/DVN/2SB7ZF 

## 3. County and State TIGER Shapefiles 
https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html 

County - level and State - level shapefiles were directly taken from the TIGER dataset, made available by the US Census Bureau, as per the 2025 boundaries. 

# Code Scripts 
The order in which the code scripts must be run is clarified through the prefix (01, 02, 03 and so forth). The first file (preprocessing.py) runs data cleaning and aggregation steps. Please find more detailed steps on the manner of replicating the data acquisition for the raw data files in 'documentation/writeup'. 

# Dashboard
The code in this repo creates a dashboard that can be found at the following link:-

This dashboard requires to be awoken after 24 hours of inactivity. This is not a bug, just an intended feature by streamlit! The dashboard showcases an interactive map of US Counties, intended for any audience interested in understanding drug abuse patterns. It covers spatial relationships across urbanization, race, sex and education levels. The interactive map can also be filtered by state to explore more granularly. Finally, the dashboard covers two metrics - Rate per 100k and Death burden. The latter is a metric of total deaths, and the former is the death count normalised by population. The most fascinating example of the differences in the use of these metrics can be found in Illinois. By rate per 100k, Marion County has the highest incidence of drug related mortality, as high as 41 percent of deaths. Comparatively, by death burden or total death count, Cook County leads not just illinois but is one of the highest in the country, with 20204 deaths from drug related mortality. Interestingly, total death count in Marion County is 15. 

This is just one example of the usefulness of this dashboard for visualisation of drug mortality in the United States. We urge the viewer to explore the interactive map for their needs, and find other trends that could inform health policy. 