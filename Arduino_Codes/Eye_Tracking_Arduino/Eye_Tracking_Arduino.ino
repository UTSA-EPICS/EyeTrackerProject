#include <Servo.h>

Servo servo_1; // Left Servo
Servo servo_2; // Right Servo
Servo servo_3; // Up Servo
Servo servo_4; // Down Servo

// Sets the input pin values
const int Left_pin_input = 8; //13
const int Right_pin_input = 12; //12
const int Up_pin_input = 13; //8
const int Down_pin_input = 7;// 7

// Sets the output pin values
const int Left_pin_output = 11;
const int Right_pin_output = 10;
const int Up_pin_output = 9;
const int Down_pin_output = 6;

// Sets the inital position values
int servo_angle_1 = 90; // Left
int servo_angle_2 = 90; // Right
int servo_angle_3 = 90; // Up
int servo_angle_4 = 90; // Down

// Test var
int test = 80;

// Bolleans used for detecting the input signals

bool Left_state = HIGH;
bool Right_state = HIGH;
bool Up_state = HIGH;
bool Down_state = HIGH;

void setup() {
  // put your setup code here, to run once:
  
  // Activates the pins 
  pinMode(Left_pin_input, INPUT);
  pinMode(Right_pin_input, INPUT);
  pinMode(Up_pin_input, INPUT);
  pinMode(Down_pin_input, INPUT);

  // Activates the pins
  servo_1.attach(Left_pin_output);
  servo_2.attach(Right_pin_output);
  servo_3.attach(Up_pin_output);
  servo_4.attach(Down_pin_output);

  Serial.begin(9300);

  Serial.println("Left State: " + Left_state);
  Serial.println("Right State: " + Right_state);
  Serial.println("Up State: " + Up_state);
  Serial.println("Down State: " + Down_state);

  servo_1.write(test);
  servo_2.write(test);
  servo_3.write(test);
  servo_4.write(test);
}

// void loop(){
//   Up_state = digitalRead(Up_pin_input);

//   if (Up_state == HIGH) {
//     Serial.println("Signal received: HIGH");
//   } else {
//         Serial.println("Signal received: LOW");

//   }
//   delay(500);
// }

void loop() {
  // put your main code here, to run repeatedly:

  // Left_state = digitalRead(Left_pin_input);
  // Right_state = digitalRead(Right_pin_input);
  // Up_state = digitalRead(Up_pin_input);
  // Down_state = digitalRead(Down_pin_input);



  // Serial.println(Left_state);
  // Serial.println(Right_state);
  // Serial.println(Up_state);
  // Serial.println(Down_state);


  if(digitalRead(Left_pin_input) == LOW){ // For Left Activation
    if (servo_angle_1 < 90){
      servo_angle_1 += 3;  // down
      servo_1.write(servo_angle_1);
      }
    if (servo_angle_2 < 150){
      servo_angle_2 += 3;  // right
      servo_2.write(servo_angle_2);
      }
    if (servo_angle_3 < 150){
      servo_angle_3 += 3;  // left
      servo_3.write(servo_angle_3);
      }
    if (servo_angle_4 < 150){
      servo_angle_4 += 3;  // up
      servo_4.write(servo_angle_4);
      }

    delay(500);
  }

  if(digitalRead(Right_pin_input) == LOW){ // For Right Activation
    if (servo_angle_2 > 90){
      servo_angle_2 -= 3;  // down
      servo_2.write(servo_angle_2);
    }
      if (servo_angle_1 > 30){
        servo_angle_1 -= 3;  // right
        servo_1.write(servo_angle_1);
        }
      if (servo_angle_3 < 150){
        servo_angle_3 += 3;  // left
        servo_3.write(servo_angle_3);
        }
      if (servo_angle_4 < 150){
        servo_angle_4 += 3;  // up
        servo_4.write(servo_angle_4);
      }
    delay(500);
  }

  if(digitalRead(Up_pin_input) == LOW){ // For Up Activation
    if (servo_angle_3 > 90){
      servo_angle_3 -= 3;  // down
      servo_3.write(servo_angle_3);
      }
      if (servo_angle_2 < 150){
        servo_angle_2 += 3;  // right
        servo_2.write(servo_angle_2);}
      if (servo_angle_1 > 30){
        servo_angle_1 -= 3;  // left
        servo_1.write(servo_angle_1);}
      if (servo_angle_4 < 150){
        servo_angle_4 += 3;  // up
        servo_4.write(servo_angle_4);
        }
    delay(500);
  }
  
  if(digitalRead(Down_pin_input) == LOW){ // For Down Activation
    if (servo_angle_4 > 90){
      servo_angle_4 -= 3;  // down
      servo_4.write(servo_angle_4);
    }
      if (servo_angle_2 < 150){
        servo_angle_2 += 3;  // right
        servo_2.write(servo_angle_2);}
      if (servo_angle_1 > 30){
        servo_angle_1 -= 3;  // left
        servo_1.write(servo_angle_1);}
      if (servo_angle_3 < 150){
        servo_angle_3 += 3;  // up
        servo_3.write(servo_angle_3);
      }
    delay(500);
    }
}
