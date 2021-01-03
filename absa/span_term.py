from collections import Counter

if __name__ == '__main__':
    target_path="../data/absa/split_rest_total_test.txt"
    count_length=[]
    span_term=[]
    with open(target_path, encoding='UTF-8-sig') as file:
        for line in file:
            cnt=0
            sent, tag_string = line.strip().split('####')  #
            word_tag_pairs = tag_string.split(' ')
            tag_list=[]
            word_list=[]
            for item in word_tag_pairs:
                # valid label is: O, T-POS, T-NEG, T-NEU
                word, tag  = item.split('=')
                tag_list.append(tag)
                word_list.append(word)
            diff=[]
            for i in range(len(tag_list)):
                if tag_list[i] !='O':
                    cnt=cnt+1
            #print(cnt)
            span_term.append(cnt)
    print(span_term.count(1))
    print(span_term.count(2))
    print(span_term.count(3))
    print(span_term.count(4))
    print(span_term.count(5))
