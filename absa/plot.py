import matplotlib.pyplot as plt
if __name__ == '__main__':
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }
    x=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0]
    y1=[68.93,68.04,66.94,66.18,67.32,65.89,
       66.44,68.08,68.51,68.37,69.13,68.29]
    y2=[74.89,75.28,75.48,75.33,75.73,75.91,75.52,76.5,75.99,
        76.59,76.41,76.15]
    y3 = [56.863, 57.447, 58.192, 59.166, 56.563, 56.705, 57.034,
          58.8, 59.276, 58.713, 59.02,57.43]
    plt.xlim(-0.01,1.01,font2)
    plt.ylim(52, 80,font2)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("F1(%)", font1,fontsize=16)
    plt.xlabel(chr(964),font1,fontsize=16)
    plt.plot(x,y1,marker="^",color='r',linestyle='--',label='Laptop')
    plt.plot(x,y2,marker="*",color='g',linestyle='--',label='Restaurant')
    plt.plot(x,y3,marker="d",color='c',linestyle='--',label='Tweets')
    ax = plt.gca()
    plt.legend(loc='lower right', prop=font2, fontsize=10, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
               ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
    #plt.savefig('lap.png')
    plt.show()

#     y2=[74.89,75.28,75.48,75.33,75.73,75.91,75.52,76.5,75.99,
#         76.59,76.41,76.15]
#     plt.xlim(-0.1,1.1)
#     plt.ylim(72, 78)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.ylabel("F1(%)", fontsize=16)
#     plt.xlabel(chr(964),fontsize=16)
#     plt.plot(x,y2,marker="s",color='r',label='res')
#     plt.savefig('res.png')
#     plt.show()
#
#     y3=[56.863,57.447,58.192,59.166,56.563,56.705,57.034,
# 58.8,59.276,58.713,59.025,57.43]
#     plt.xlim(-0.1,1.1)
#     plt.ylim(52, 62)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.ylabel("F1(%)", fontsize=16)
#     plt.xlabel(chr(964),fontsize=16)
#     plt.plot(x,y3,marker="d",color='m',label='twi')
#     plt.savefig('twi.png')
#     plt.show()

