package com.example.carapp.Controller.CallBacks;

import com.example.carapp.Entites.Car;
import com.example.carapp.Entites.UserType;

public interface SelectCarCallBack {

    void onSuccess(Car car);
    void onFailer(String error);

}
