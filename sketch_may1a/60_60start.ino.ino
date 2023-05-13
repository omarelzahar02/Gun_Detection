#include<Servo.h>
Servo ser1;
Servo ser2;
Servo ser3;

void setup() {
 ser1.attach(A0);//the first servo is attached to A0 which is the X axis servo
 ser2.attach(A1);//the second servo is attached to A1 which is the Y axis servo
 ser3.attach(A2);//the third servo is attached to A2 which is the trigger servo
 ser3.write(0);
 Serial.begin(9600);
 
  ser2.write(60);
  ser1.write(60);
}

void loop(){



  
}