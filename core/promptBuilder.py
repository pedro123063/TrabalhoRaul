import numpy as np

def prompt_template():
    prompt={}
    prompt[0]=f"Você irá prever o valor de uma ação."
    prompt[1]=f"Você receberá conjuntos de "
    prompt[2]=f" valores de ultimos dias "
    prompt[3]=f" juntos de seus gabaritos para treinar e tentar adivinhar o próximo."
    prompt[4]=f" para adivinhar o próximo."
    prompt[5]=f"Os valores são: "
    prompt[6]=f"."
    prompt[7]=f"Qual o próximo valor?Responda apenas com o número previsto."
    return prompt

def assembleZeroShot(data_array,study_window):
    study_pack=data_array[-study_window-1:-1]
    p=prompt_template()
    string_study_pack=""
    for i in range(len(study_pack)):
        if i< len(study_pack)-1:
            string_study_pack+=f"{study_pack[i]} , "
        else:
            string_study_pack+=f"{study_pack[i]} "

    my_string=p[0]+p[1]+f"{study_window}"+p[2]+p[4]+p[5]+string_study_pack+p[6]+p[7]
    return my_string

def assembleFewShot(few_shot_data, study_window, number_of_studies):
    p = prompt_template()
    my_string = p[0] + p[1] + f"{study_window}" + p[2] + p[3] + "\n\n"
    for k in range(number_of_studies):
        inicio = k * (study_window + 1)
        fim_janela = inicio + study_window
        
        aux = few_shot_data[inicio:fim_janela]
        gab = few_shot_data[fim_janela]
        
        string_aux = f"Exemplo {k+1}: "
        for i in range(len(aux)):
            string_aux += f"{aux[i]} , "
        string_aux += f"Gabarito: {gab}\n"
        
        my_string += string_aux

    study_pack = few_shot_data[-study_window:]
    string_study_pack = ""
    for i in range(len(study_pack)):
        if i < len(study_pack) - 1:
            string_study_pack += f"{study_pack[i]} , "
        else:
            string_study_pack += f"{study_pack[i]}"

    my_string += "\n" + p[5] + string_study_pack + p[6] + " " + p[7]
    
    return my_string