import pandas as pd
import numpy as np

# List of file to be processed
files = ['datatraining', 'datatest', 'datatest2']


def is_constrained(i,idx):
    if i==1 and idx == 0: # Base Constraints
        return True
    elif i==3 and idx<=2: # Base Constraints
        return True
    elif i==5 and idx<=4: # Base Constraints
        return True
    return False


def preprocess():
    for file in files:
        # Read Data
        data = pd.read_csv('data/%s.txt' % file)

        # Dropping unnecessary index
        data = data.drop(['index'], axis=1)

        # Processing date
        hours = []
        minutes = []
        seconds = []
        for d in data['date']:
            t = d.split(' ')[1].split(':')
            hours.append(int(t[0]))
            if int(t[2]) == 0: # If it is exactly beginning of the minute, assign it as is.
                minutes.append(int(t[1]))
            else: # Else ceil it.
                minutes.append(int(t[1])+1)
        data['time/hour'] = hours
        data['time/minute'] = minutes

        # Change Loop
        intervals = [1,3,5]
        for i in intervals:

            Humidity = [0]*i
            CO2 = [0]*i
            Light = [0]*i
            Temperature = [0]*i
            
            # Humidity change
            for idx, h in enumerate(data['Humidity']):
                constrained = is_constrained(i,idx)
                if constrained:
                    continue
                Humidity.append(h-data['Humidity'][idx-i])
            h_field = 'Humiditiy_d' + str(i)
            data[h_field] = Humidity 

            # CO2 change
            for idx, c in enumerate(data['CO2']):
                constrained = is_constrained(i,idx)
                if constrained:
                    continue
                CO2.append(c - data['CO2'][idx - i])
            c_field = 'CO2_d' + str(i)
            data[c_field] = CO2

            # Light change
            for idx, l in enumerate(data['Light']):
                constrained = is_constrained(i,idx)
                if constrained:
                    continue
                Light.append(l - data['Light'][idx - i])
            l_field = 'Light_d' + str(i)
            data[l_field] = Light

            # Temperature change
            for idx, t in enumerate(data['Temperature']):
                constrained = is_constrained(i,idx)
                if constrained:
                    continue
                Temperature.append(t - data['Temperature'][idx - i])
            t_field = 'Temperature_d' + str(i)
            data[t_field] = Temperature

        # Drop Date
        data = data.drop('date', axis=1)

        # Save new CSV
        data.to_csv('processed_data/%s.csv' % file, encoding='utf-8', index=False)
        print('\t\tSaved %s.csv' % file)

if __name__=='__main__':
    preprocess()