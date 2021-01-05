
def competitive_matching(sorted_list):
    aligned=[]
    source=[]
    target=[]

    aligned_2 = []
    source_2 = []
    target_2 = []

    aligned_3 = []
    source_3 = []
    target_3 = []

    for pair in sorted_list.keys():
        if pair[0] not in source and pair[1] not in target:
            aligned.append(pair)
            source.append(pair[0])
            target.append(pair[1])
        elif pair[0] not in source_2 and pair[1] not in target_2:
            aligned_2.append(pair)
            source_2.append(pair[0])
            target_2.append(pair[1])
        elif pair[0] not in source_3 and pair[1] not in target_3:
            aligned_3.append(pair)
            source_3.append(pair[0])
            target_3.append(pair[1])
    return [aligned, aligned_2, aligned_3]

def best_matching(sorted_list):
    source = {}
    target = {}

    for pair in sorted_list.keys():
        val = sorted_list[pair]
        if(pair[0] in source):
            temp = source[pair[0]]
            if(val<temp[1]):
               source[pair[0]] = [pair[1],val]
        else:
            source[pair[0]] = [pair[1],val]
        
        if(pair[1] in target):
            temp = target[pair[1]]
            if(val<temp[1]):
               target[pair[1]] = [pair[0],val]
        else:
            target[pair[1]] = [pair[0],val]
    return [source,target]
    

def best_matching_2(sorted_list):
    source = {}
    target = {}

    for pair in sorted_list.keys():
        val = sorted_list[pair]
        if(pair[0] in source):
            temp = source[pair[0]]
            if(len(temp)<5):
               temp.append([pair[1],val])
               source[pair[0]] = temp
        else:
            source[pair[0]] = [[pair[1],val]]
        
        if(pair[1] in target):
            temp = target[pair[1]]
            if(len(temp)<5):
               temp.append([pair[0],val])
               target[pair[1]] = temp 
        else:
            target[pair[1]] = [[pair[0],val]]
    return [source,target]

def best_matching_3(sorted_list, threshold):
    source = {}

    for pair in sorted_list.keys():

        if(pair[0] not in source):
            source[pair[0]] = []

        val = sorted_list[pair]
        if (val <= threshold):
                temp = source[pair[0]]
                temp.append([pair[1],val])
                source[pair[0]] = temp

    return source