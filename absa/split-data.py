
if __name__ == '__main__':
    root_path="../data/absa/twitter10_train.txt"
    target_path="../data/absa/split_twitter10_train.txt"
    test_result=open(target_path,"w",encoding='utf-8')
    length_list=[]
    with open(root_path, encoding='UTF-8-sig') as file:
        for line in file:
            record = {}
            sent, tag_string = line.strip().split('####')  #
            record['sentence'] = sent
            word_tag_pairs = tag_string.split(' ')
            sent=sent.split(' ')
            # if len(word_tag_pairs)>=60:
            #     word_tag_pairs=word_tag_pairs[0:59]
            #     sent=sent[0:59]
            #print(word_tag_pairs)  # 1  ['Sad=T_POS', 'very=O', 'SAD=O', '.=O']
            tag_list=[]
            word_list=[]
            for item in word_tag_pairs:
                # valid label is: O, T-POS, T-NEG, T-NEU
                eles  = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                elif len(eles) > 2:
                    tag = eles[-1]
                    word = (len(eles) - 2) * "="
                tag_list.append(tag)
                word_list.append(word)
            #print(tag_list)  #2['T_POS', 'O', 'O', 'O']
            cnt=0
            diff=[]
            for i in range(len(tag_list)):
                diff_tag = ['O' for p in range(len(tag_list))]
                if tag_list[i] !='O':
                    for j in range(i,len(tag_list)):
                        if tag_list[j]=='O' or j== len(tag_list)-1:
                            cnt=cnt+1
                            for k in range(i,j+1):
                                t = tag_list[k]
                                diff_tag[k] = t
                                tag_list[k] = 'O'
                            diff.append(diff_tag)
                            break
            if diff==[]:
                diff = [['O' for p in range(len(tag_list))]]
            #print(word_list)  # 3 ['Sad', 'very', 'SAD', '.']
            #print(diff)   # 4 [['T_POS', 'O', 'O', 'O']]
            length_list.append(cnt)
            #print(cnt)
            later_string=[]
            for p in range(len(diff)):
                temp=[]
                for q in range(len(word_list)):
                    str1 = word_list[q]+'='+diff[p][q]
                    temp.append(str1)
                later_string.append(temp)
            #print(later_string)  # 5  [['Sad=O', 'very=O', 'SAD=O', '.=O']]
            for p in range(len(diff)):
                if diff != [['O' for p in range(len(tag_list))]]:  #
                    stenence=' '.join(sent)+"####"+' '.join(later_string[p])
                    test_result.write("{}\n".format(stenence))
                    print(stenence)
    test_result.close()


