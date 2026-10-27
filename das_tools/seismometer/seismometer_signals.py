

from obspy import read

def read_seismometer_data(channels,path_,date,st):
    # print(date.julday)
    for chan in channels:    
        new_path = path_+'/'+chan+'.D/*'+str(date.julday)
        tr = read(new_path)
        for i in range(0,len(tr)):
            st.append(tr[i])
    return st
