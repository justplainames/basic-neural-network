#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>


//****************************************************************************************************************************************************//
double lr_calculator(double dataset[], double weight[], double bias, int n);
double activator(double linear_regression);
double mae_calculator(double activator, double p_output);
double absmae_calculator(double activator, double p_output);
double mmse_calc(double mae[], double dataset[100][10], int count);
double output_error(double mae[], double linear_regression[], int count);
void hidden_error(double error, double weight[], double neuron_output[][6], double *ptr);
void output_propogation(double learning_rate,double sigma_error,double *ptr_weight, double *ptr_bias);
void hidden_propogation(double learning_rate,double *ptr_error, double *ptr_weight, double *ptr_bias);
double randnum();
//****************************************************************************************************************************************************//



int main(){

    clock_t begin = clock();                                                    //Start timing for program run


    FILE* filepointer;                                                          //Initialize file pointer filepointer
    FILE* end_result;                                                           //Initialize file pointer end_result                                                                 

    char buffer[50], temp[101][50]; 
    int counter = 0;

    double train_data[91][10], train_activation[91], train_hidden_activation[91][6], train_linear_regression[91],train_neuron_output[91][6];
    double test_data[10][10], test_activation[11],test_hidden_activation[11][6], test_linear_regression[11], test_neuron_output[11][6];


    double output_weight[6], hidden_weight[6][9],hidden_bias[6], output_bias, learning_rate = 0.05, mae_bench = 0.25, train_mae[101],train_abs_mae[101];
    double test_mae[10], test_abs_mae[10], hidden_sigma_error[6];

    
    int flag = 0, iteration_count = 0;                                         //Initialize flag = 0 and iteration count = 0
//----------------------------------------------------------------------------------------------------------------------------------------------
fopen_s(&filepointer, "fertility_Diagnosis_Data_Group9_13.txt", "r");           //Opens the txt file "fertility_Diagnosis_data_Group9_13.txt" of data for reading
    if(filepointer == NULL){ 
        printf("\nNo file found");                                              //If no file is found, program will end
        exit(1);
    }

    while(fgets(buffer, 256, filepointer) != NULL){                             //While loop to go through each line in the text file
        sscanf(buffer, "%s", &temp[counter]);                                   //Adds each line(patient data) into an array called temp, position based on counter 
        counter++;                                                              //+1 to variable called counter after every iteration       
    
    }                                                                
    fclose(filepointer);                                                        //Closes the patient data file


    for(int i = 0; i<100; i++){                                                 //Loops thorough all 100 patient data
        char *x = strtok(temp[i], ",");                                         //goes through each char till a "," is found and replaces it wth \0. The string of char stores it's address into pointer x
        int j = 0;                                                              //J is used to point to the correct column(each attribute)
        while(x != NULL){                                                       //While loop, ends when x = NULL. 
            if(i<90){                                                           //Goes through the first 90 patients for training
                train_data[i][j] = atof(x);                                     //Converts x into a float and stores in a array for training
                x = strtok(NULL, ",");                                          //Takes the next attribute and store it's address into x
                j++;                                                            
            }               
            else{                                                               //After the 90 patients, the next 10 is stored for testing
                test_data[i-90][j] = atof(x);   
                x = strtok(NULL, ",");
                j++;  
            }
        }
    }

    output_bias = randnum();                                                    //Uses the randum function to store a random number for the bias
    for(int i=0; i<6; i++){                                                     //Loops 6 times
        hidden_bias[i] = randnum();                                             //Each loop assigns calls randnum function and assigns 6 different biases for the hidden layer 
    }
    for(int i=0; i<6; i++){                                                     //Loops through create 6 different numbers to create 6 random weights
        output_weight[i] = randnum();                                           //Calls the randum function to generate a random weight
    }
    for(int i=0; i<6; i++){                                                     //Loops through each neuron
        for(int j=0; j<9; j++){                                                 //Within each neuron, loops through each input
            hidden_weight[i][j] = randnum();                                    //Calls the randum function to generate a random weight(54 times)
        }
    }

 //------------------------------------------------------------------------------------------------------------------------------------------------------------------//
    fopen_s(&end_result, "end_result.txt", "w");                                //Create/open a text file called end_result in write mode to overwrite the text file with the latest data that runs through the feedforward and backpropagation
    if(filepointer == NULL){                                                    //If no file is found, program will end
        printf("\nFile could not be opened"); 
        exit(1);
    }

    //MMSE CALCULATION PRE TRAINING//

    for(int i = 0; i<10; i++){                                                                                  //Loops through each  patient in the testing data                                   
        for(int j = 0; j<6; j++){                                                                               //Loops through each neuron in the hidden layer
            test_neuron_output[i][j] = lr_calculator(test_data[i], hidden_weight[j], hidden_bias[j], 9);        //Call lr_calculator function to calculate linear regression of patient per neuron
            test_hidden_activation[i][j] = activator(test_neuron_output[i][j]);                                 //Call activator dunction to calculate sigma activation output of patient per neuron
        }
        test_linear_regression[i] = lr_calculator(test_hidden_activation[i], output_weight, output_bias, 6);    //Call lr_calculator function to calculate linear regression using data from the hidden layer
        test_activation[i] = activator(test_linear_regression[i]);                                              //Call activator function to calculate linear sigma activation output using data from the hidden layer
        test_mae[i] = mae_calculator(test_activation[i], test_data[i][9]);                                      //Call mae_calculator funciton to calculate mae
        test_abs_mae[i] = absmae_calculator(test_activation[i], test_data[i][9]);                               //Call absmae_calculator function to caluclate absolute mae
    }

    //ABOVE IS REPEATED BELOW BUT ON THE TRAINING DATA//
    
    for(int i = 0; i<90; i++){
        for(int j=0; j<6; j++){
            train_neuron_output[i][j] = lr_calculator(train_data[i], hidden_weight[j], hidden_bias[j], 9);
            train_hidden_activation[i][j] = activator(train_neuron_output[i][j]);
        }
        train_linear_regression[i] = lr_calculator(train_hidden_activation[i], output_weight, output_bias, 6);
        train_activation[i] = activator(train_linear_regression[i]);
        train_mae[i] = mae_calculator(train_activation[i], train_data[i][9]);
        train_abs_mae[i] = absmae_calculator(train_activation[i], train_data[i][9]);
    
    }

    
    double pre_test = mmse_calc(train_mae, train_data, 90);                                                     //Calls the mmse_calc function to calculate mae for the training data
    double pre_train = mmse_calc(test_mae, test_data, 10);                                                      //Calls the mmse_calc function to calculate mae for the testing data
    printf("\nPre-Training MMSE for Training Data: %f", pre_train);
    printf("\nPre-Training MMSE for Testing Data: %f", pre_test);

    while(flag == 0){                                                                                           //Will loop till the flag turns 1(indicating results pass the mae bench)
        double total_mae = 0;                                                                                   //Reinitialize total mae to 0
 
        iteration_count += 1;                                                                                   //Keeps track on how many iterations were completed
        for(int i = 0; i<90; i++){                                                                              //Goes through each patient
            for(int j = 0; j<6; j++){                                                                           //Goes through each neuron in the hidden layer
                train_neuron_output[i][j] = lr_calculator(train_data[i], hidden_weight[j], hidden_bias[j],9);   //Call lr_calculator function to calculate linear regression of patient per neuron and places in an array
                train_hidden_activation[i][j] = activator(train_neuron_output[i][j]);                           //Call activator function to calculate linear sigma activation output using data from the hidden layer and places in an array
            }

            train_linear_regression[i] = lr_calculator(train_hidden_activation[i], output_weight, output_bias, 6);  //Call lr_calculator function to calculate linear regression using data from the hidden layer and places in an array
            train_activation[i] = activator(train_linear_regression[i]);                                            //Call activator function to calculate linear sigma activation output using data from the hidden layer and places in an array
            train_mae[i] = mae_calculator(train_activation[i], train_data[i][9]);                                   //Call mae_calculator funciton to calculate mae
            train_abs_mae[i] = absmae_calculator(train_activation[i], train_data[i][9]);                            //Call absmae_calculator function to caluclate absolute mae and place result in an array
            total_mae += train_abs_mae[i];
        }
     
        total_mae = total_mae/90.0;
    
        fprintf(end_result, "%d\t %lf\n", iteration_count, total_mae);                                             //Write the iteration and the mae of each patient into the end_result.txt file 
        //printf("\nItertation count = %d\t total_mae = %f", iteration_count, total_mae);

        if(total_mae > mae_bench){                                                                                 //If statement to check if the guess error margin is below the threshold
            double sigma_error = output_error(train_mae, train_linear_regression, 90);                             //Calls function output_error to calculate error of neuron in output layer
            hidden_error(sigma_error, output_weight, train_neuron_output, &hidden_sigma_error);                    //Calls function hidden_error to calculate error of neuron in hidden layer of each neuron 
            output_propogation(learning_rate, sigma_error, &output_weight, &output_bias);                          //Calls function output_propagation to correct both the bias and weight between the hidden 
            hidden_propogation(learning_rate, &hidden_sigma_error, &hidden_weight, &hidden_bias);                  //Calls function hidden_propagation to correct both the weight bias between the input and hidden layer
        }

        else                                                                                                        //If benchmark passes flag chages to 1 to end WHILE loop
        {
            flag = 1;
        }        

    }
    fclose(end_result);                                                                                             //Closes the txt file "end_result.txt"

    //CALCULATE MMSE FOR BOTH TRAINING AND TESTING DATA FOR AFTER TRAINING
    for(int i = 0; i<10; i++){                                                                                      
        for(int j = 0; j<6; j++){
            test_neuron_output[i][j] = lr_calculator(test_data[i], hidden_weight[j], hidden_bias[j], 9);
            test_hidden_activation[i][j] = activator(test_neuron_output[i][j]);
        }
        test_linear_regression[i] = lr_calculator(test_hidden_activation[i], output_weight, output_bias, 6);
        test_activation[i] = activator(test_linear_regression[i]);
        test_mae[i] = mae_calculator(test_activation[i], test_data[i][9]);
        test_abs_mae[i] = absmae_calculator(test_activation[i], test_data[i][9]);
        }

    
    double post_test = mmse_calc(train_mae, train_data, 90);
    double post_train = mmse_calc(test_mae, test_data, 10);
    printf("\nPost-Training MMSE for Training data: %f", post_train);
    printf("\nPost-Training MMSE for Testing data: %f", post_test);
    

    int tp = 0, tn = 0, fp = 0, fn = 0;                                                                             //Declare and initialize the 4 variables, True negative and output and False negative and output to 0
    for(int i = 0; i<90; i++){                                                                                      //Loops through the result for each patient
        double temp = train_data[i][9];                                                                             //Assign a variable called temp and store the actual result of the patient
        int result;
        result = (int)temp;                                                                                         //Changes temp from double data type to integer data type

        // This section does the confusion matrix. It compares the actual result with the guess(result >= 0.5 is considered a 1 while <0.5 is considered a 0) by the program to determine if it is a false negative, positive or true negative, positive. 
        if((result == 0) && (test_activation[i] >= 0.5)){                                                           
            fp += 1;
        }
        else if((result == 0) && (test_activation[i] < 0.5)){
            tn += 1;
        }
        else if((result == 1) && (test_activation[i] >= 0.5)){
            tp += 1;
        }
        else if((result == 1) && (test_activation[i] < 0.5)){
            fn += 1;
        }

        else{printf("\nERROR!");}

    }

    //PRINTS CONFUSION MATRIX TABLE
    printf("\n           TESTING DATA SET   ");
    printf("\n           ___________________");
    printf("\n          |  True  |  False   |");
    printf("\n _________|________|__________|");
    printf("\n|Positive |    %d   |    %d    |", tp,fp);
    printf("\n|---------|--------|----------|");
    printf("\n|Negative |    %d  |    %d    |", tn, fn);
    printf("\n|-----------------------------|") ;
    

    tp = 0, tn = 0, fp = 0, fn = 0;                                                                                 //Resets the count to 0
    for(int i = 0; i<10; i++){                                                                                      //Loops through the result for each patient
        double temp = test_data[i][9];                                                                             //Assign a variable called temp the actual result of the patient
        int result;
        result = (int)temp;                                                                                         //Changes temp from float data type to integer data type

        // SAME AS ABOVE BUT IS USED FOR THE TESTING SET
        if((result == 0) && (test_activation[i] >= 0.5)){                                                           
            fp += 1;
        }
        else if((result == 0) && (test_activation[i] < 0.5)){
            tn += 1;
        }
        else if((result == 1) && (test_activation[i] >= 0.5)){
            tp += 1;
        }
        else if((result == 1) && (test_activation[i] < 0.5)){
            fn += 1;
        }

        else{printf("\nERROR!");}

    }

    //PRINTS CONFUSION MATRIX TABLE
    printf("\n           TRAINING DATA SET  ");
    printf("\n           ___________________");
    printf("\n          |  True  |  False   |");
    printf("\n _________|________|__________|");
    printf("\n|Positive |   %d   |    %d    |", tp,fp);
    printf("\n|---------|--------|----------|");
    printf("\n|Negative |   %d   |    %d    |", tn, fn);
    printf("\n|-----------------------------|") ;





    system("gnuplot -p -e \"plot 'end_result.txt'\"");                                                              //opens GNU plot    

    clock_t end = clock();                                                                                          //Record time ended    
    double time_spent = ((double)(end - begin)) / CLOCKS_PER_SEC;                                                   //calculates the time taken for the program to run 
    printf("\n\nTime elpased is %f seconds \nTotal iteration count = %d", time_spent, iteration_count);             //Prints out program run time and total iteration count

    return 0;


}



//************************************************************************FUNCTIONS************************************************************************************//


double randnum(){                                                                                                  // Generates and return random number
    srand((unsigned int)time(NULL));
    return((float)rand())/(RAND_MAX*2);
}

double lr_calculator(double dataset[], double weight[],double bias, int n){                                        //Calculates and return linear regression uses an array of a patient data set, weight, bias and count for number of data as arguments
    double temp_lr = 0;
    for(int i = 0; i < n; i++){
            temp_lr += weight[i]*dataset[i];       
    }
    temp_lr += bias;
    return temp_lr;
}

double activator(double linear_regression){                                                                        //Calculate and return sigmoid activation, takes in the linear regression as the argument
    return 1/(1+exp(-1*linear_regression));
}

double mae_calculator(double activator, double p_output){                                                          //Calculates and reutrn mae, takes sigmoid activation and actual output
    return (activator - p_output);
}

double absmae_calculator(double activator, double p_output){                                                       //Calculates and reutrn absolute value of the mae takes in the sigmoid activation and actua out as arguments 
    return fabs(activator - p_output);  
}

double output_error(double mae[], double linear_regression[], int count){                                          //Calculates and reutrn the error for the neuron in the output layer, takes in the mae, linear regression and data size       
    double sigma_error = 0;
    for(int i = 0; i<90; i++){
        sigma_error += (mae[i]) * ((exp(linear_regression[i]))/(pow((1+exp(linear_regression[i])),2)));
    }
    return sigma_error = sigma_error/90.0;
}

void hidden_error(double error, double weight[], double neuron_output[][6], double *ptr){                          //Calculates and updates the error for the neurons in the hidden layer, takes in the error of the output, weight in the output and the linear regresion of the output
    for(int i = 0; i<6; i++){                  
        double temp_total_error = 0;
        for(int j=0; j<90; j++){    
            temp_total_error += error * weight[i] * (exp(neuron_output[j][i]))/(pow((1+exp(neuron_output[j][i])),2));
        }
        *(ptr+i) = temp_total_error / 90.0;
    }
}

void output_propogation(double learning_rate,double sigma_error,double *ptr_weight, double *ptr_bias){              //Updates the weight and bias between the hidden layer and output layer, takes in learning rate, the output error, address of the weight and bias between the hidden and output layer
    for(int i = 0; i<6; i++){
        *(ptr_weight+i) = *(ptr_weight+i) - (learning_rate*sigma_error);
    }
    *ptr_bias = *ptr_bias -  (learning_rate * sigma_error);
}


void hidden_propogation(double learning_rate, double *ptr_error, double *ptr_weight, double *ptr_bias){             //Updates the weight and bias between the input and hidden layer, takes in learning rate, the output error, address of the weight and bias between the input and hidden layer
    for(int i = 0; i<6; i++){
        for(int j = 0; j<9; j++){
            *ptr_weight = *ptr_weight - (learning_rate * *(ptr_error+i));
            ++ptr_weight;
        }
        *ptr_bias = *ptr_bias - (learning_rate* ptr_error[i]);
    }
}

double mmse_calc(double mae[], double dataset[100][10], int count){                                                 //Caluclate and return the mmse, takes in the arguments of the array of mae, patient data and data size
    double mmse = 0;
    for(int i = 0; i<count; i++){
        mmse += pow((mae[i] - dataset[i][9]),2);
    }
    mmse = mmse/count;
    return mmse;
}

//

/*

PSEDUOCODe

BEGIN
    OPEN FILE "fertility_Diagnosis_Data_Group9_13.txt"
    IF FILE == NULL do
        exit
    END IF

    WHILE FILE != NULL do
        READ temp[i]
        i <- i+1
    END WHILE
     i <- 0
    CLOSE FILE

    FOR i=0 to i=99 do
        ptr_x <- strtok(temp, ",")
        j <- 0
        WHILE ptr_x != NULL
            IF i<90 do
                train_data[i][j] <- atof(x)
                x <- strtok(NULL, ",")
                j <- j+1
            END IF
            ELSE
                train_data[i-90][j] <- atof(x)
                x <- strtok(NULL, ",")
                j <- j+1
            END ELSE
        END WHILE
    END FOR

    output_bias <- randnum()
    FOR i=0 to i = 5 do
        hidden_bias[i] <- randnum()
        i <- i+1
    END FOR 
    FOR i=0 to i = 9 do
        output_weight[i] <- randnum()
        i <- i+1
    END FOR
    FOR i=0 to i = 5 do
        FOR j=0 to j = 9 do
            hidden_weight[i][j] <- randnum()
            j <- j+1
        END FOR
        i <- i + 1
    END FOR

    OPEN FILE "endresult.txt"
    IF FILE == NULL do
        exit
    END IF

    FOR i = 0 to i = 9 do
        FOR j = 0 to j = 5 do
            test_neuron_output[i][j] <- lr_calculator(test_data, hidden_weight, hidden_bias, 9)
            test_hidden_activation[i][j] = activator(test_neuron_output)
            j <- j + 1
        END FOR
        test_linear_regression[i] = lr_calculator(test_hidden_activation, output_weight, output_bias, 6)
        test_activation[i] = activator(test_linear_regression)
        test_mae[i] = mae_calculator(test_activation, test_data)
        test_abs_mae[i] = absmae_calculator(test_activation, test_data)
        i <- i + 1
    END FOR

    FOR i = 0 to i = 89 do
        FOR j = 0 to j = 5 do
            train_neuron_output[i][j] <- lr_calculator(train_data, hidden_weight, hidden_bias, 9)
            train_hidden_activation[i][j] = activator(train_neuron_output);
            j <- j + 1
        END FOR
        train_linear_regression[i] = lr_calculator(train_hidden_activation, output_weight, output_bias, 6)
        train_activation[i] = activator(train_linear_regression)
        train_mae[i] = mae_calculator(train_activation, train_data)
        train_abs_mae[i] = absmae_calculator(train_activation, train_data)
        i <- i + 1
    END FOR

    pre_test <- mmse_calc(train_mae, train_data, 90);
    pre_train <- mmse_calc(test_mae, test_data, 10);
    print "Pre-Training MMSE for Training Data", pre_train
    print "Pre-Training MMSE for Testing Data", pre_test

    WHILE flag == 0 do
        total_mae <- 0
        iteration count <- iteration count + 1
        FOR i = 0 to i = 89 do
            FOR j = 0 to j = 5 do
                train_neuron_output[i][j] <- lr_calculator(train_data, hidden_weight, hidden_bias, 9)
                train_hidden_activation[i][j] = activator(train_neuron_output);
                j <- j + 1
            END FOR
            train_linear_regression[i] = lr_calculator(train_hidden_activation, output_weight, output_bias, 6)
            train_activation[i] = activator(train_linear_regression)
            train_mae[i] = mae_calculator(train_activation, train_data)
            train_abs_mae[i] = absmae_calculator(train_activation, train_data)
            total_mae <- total_mae + train_abs_mae
            i <- i + 1
        END FOR

        total_mae <- total_mae/90.0
        FILEPRINT "endresult" , iteration_count, total_mae

        IF total_mae > mae_bench do
            sigma_error <- output_error(train_mae, train_linear_regression, 90)
            hidden_error(sigma_error, output_weight, train_neuron_output, &hidden_sigma_error)
            output_propogation(learning_rate, sigma_error, &output_weight, &output_bias)
            hidden_propogation(learning_rate, &hidden_sigma_error, &hidden_weight, &hidden_bias)
        END IF
        ELSE
            flag <- 1
        END ELSE

    CLOSE FILE "end_result"

    FOR i = 0 to i = 9 do
        FOR j = 0 to j = 5 do
            test_neuron_output[i][j] <- lr_calculator(test_data, hidden_weight, hidden_bias, 9)
            test_hidden_activation[i][j] = activator(test_neuron_output)
            j <- j + 1
        END FOR
        test_linear_regression[i] = lr_calculator(test_hidden_activation, output_weight, output_bias, 6)
        test_activation[i] = activator(test_linear_regression)
        test_mae[i] = mae_calculator(test_activation, test_data)
        test_abs_mae[i] = absmae_calculator(test_activation, test_data)
        i <- i + 1
    END FOR
    post_test <- mmse_calc(train_mae, train_data, 90);
    post_train <- mmse_calc(test_mae, test_data, 10);
    print "Post-Training MMSE for Training Data", post_train
    print "Post-Training MMSE for Testing Data", post_test

    FOR i = 0 to i = 9 do
        temp <- test_data[i][9]
        IF temp == 0 && test_activation[i] >= 0.5 do
            fp <- fp + 1
        END IF
        ELSE IF temp == 0 && test_activation[i] < 0.5 do
            tn <- tn + 1
        END ELSE IF
        ELSE IF temp == 1 && test_activation[i] >= 0.5 do
            tp <- tp + 1
        END ELSE IF    
        ELSE IF temp == 1 && test_activation[i] < 0.5 do
            fn <- fn + 1
        END ELSE IF

        ELSE 
            print "ERROR!"

        END ELSE
END

FUNCTION randnum()
    srand((unsigned int)time(NULL))
    return ((float)rand())/(RAND_MAX*2)
END FUNCTION

FUNCTION lr_calculator(dataset,weight,bias, n)
    double temp_lr <- 0
    FOR i = 0 to N-1
        temp_lr <- templr + weight[i]*dataset[i]
        i <- i + 1
    END FOR

    temp_lr <- temp_lr + bias
    return temp_lr
END FUNCTION

FUNCTION activator(linear_regression)
    return 1/(1+exp(-1*linear_regression))
END FUNCTION

FUNCTION mae_calculator(activator, p_output)
    return (activator - p_output)
END FUNCTION

FUNCTION absmae_calculator(activator, p_output)
    return fabs(activator - p_output)
END FUNCTION

FUNCTION output_error(mae, linear_regression, count)
    sigma_error <- 0
    FOR i = 0 to i = 89 do
        sigma_error <- sigma_error + ((mae[i]) * ((exp(linear_regression[i]))/(pow((1+exp(linear_regression[i])),2)))
        i <- i + 1
    END FOR
    return sigma_error = sigma_error/90.0;
END FUNCTION

FUNCTION output_propogation(learning_rate, sigma_error,ptr_weight, ptr_bias)
    ptr_weight referTodouble -> &output_weight
    ptr_bias referTodouble -> &output_bias
    FOR i = 0 to i = 5 do
        *(ptr_weight+i) <- *(ptr_weight+i) - (learning_rate*sigma_error)
        i <- i + 1
    END FOR
    *ptr_bias <- *ptr_bias -  (learning_rate * sigma_error)
END FUNCTION

FUNCTION hidden_propogation(learning_rate,ptr_error,ptr_weight,ptr_bias){
    ptr_weight referTodouble -> &hidden_weight
    ptr_bias referTodouble -> &hidden_bias
    ptr_error referTodouble -> &hidden_sigma_error
    FOR i = 0 to i = 5 do
        FOR j = 0 to j = 8 do
            *ptr_weight <- *ptr_weight - (learning_rate * *(ptr_error+i))
            ++ptr_weight
            j <- j + 1
        END FOR
        *ptr_bias = *ptr_bias - (learning_rate* ptr_error[i])
        i <- i + 1
    END FOR
END FUNCTION

FUNCTION mmse_calc(mae,dataset,count){
    mmse <- 0;
    FOR i = 0 to count -1
        mmse <- mmse + pow((mae[i] - dataset[i][9]),2)
        i <- i + 1
    END FOR
    mmse <- mmse/count;
    return mmse;
END FUNCTION

FUNCTION hidden_error(error,weight,neuron_output,ptr){
    ptr referToDouble -> &hidden_sigma_error
    FOR i = 0 to i = 5 
        temp_total_error <- 0
        FOR j = 0 to j = 89    
            temp_total_error <- temp_total_error + (error * weight[i] * (exp(neuron_output[j][i]))/(pow((1+exp(neuron_output[j][i])),2)))
            j <- j+1
        END FOR
        
        *(ptr+i) = temp_total_error / 90.0;
        i <- i + 1
    END FOR 
END FUNCTION
*/