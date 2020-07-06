#define RF 5 //L298n Motor Driver pins.   //  LF /\  /\ RF
#define RB 6                              //    |      |
#define LF 10                             //    |      |
#define LB 11                             //  LB \/  \/ RB

int Direction;                            //Int to store the input signal from Rx.
int LastDirection;            


// speed must be in the range of  0 - 255. as it is serial communication of 8 bits 
int speedup=150;
int Speed = 100; 
int slowdown =50;

void setup() {
//intializ GPIO Pins 
  pinMode(RF, OUTPUT);
  pinMode(RB, OUTPUT);
  pinMode(LF, OUTPUT);
  pinMode(LB, OUTPUT);

  Serial.begin(9600);                     //Set the baud rate to your Bluetooth module.
}

void loop() {
//  Forward();                     //just for check the anomaly detection
  delay(200);
  if (Serial.available() > 0) {
    Direction = Serial.read();
    if(LastDirection!=Direction)
    {
      Stop();
      delay(200);
    }
    switch (Direction) {
      case 'F':
        Forward();
        break;
      case 'L':
        Leftt();
        break;
      case 'R':
        Rightt();
        break;
      case 'S':
        Stop();
        break;
    }
  }
  LastDirection=Direction;
}
void Forward(){
  analogWrite(RF, Speed+15);
  analogWrite(LF, Speed);
}
void Leftt(){
  analogWrite(LF, speedup);
  analogWrite(RF, slowdown+15);
}
void Rightt(){
  analogWrite(LF, slowdown);
  analogWrite(RF, speedup+15);
}
void Stop()  {
  analogWrite(RF, 0);
  analogWrite(RB, 0);
  analogWrite(LF, 0);
  analogWrite(LB, 0);
}