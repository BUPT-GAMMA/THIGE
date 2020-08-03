from datetime import datetime, timedelta


def time_to_stamp(str_time, time_formation='%Y-%m-%d %H:%M:%S'):
    t1 = datetime.timestamp(datetime.strptime(str_time, time_formation))
    return t1

def stamp_to_time(tvalue, time_formation='%Y-%m-%d %H:%M:%S'):
    t1 = datetime.fromtimestamp(tvalue)
    str_time = datetime.strftime(t1, time_formation)
    return str_time

if __name__=="__main__":

    print(time_to_stamp('2017-12-03 00:00:00'))
    print(stamp_to_time(1512101532))

