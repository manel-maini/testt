package com.example.myshite.App.Database;

public class Object {

    private int id;
    private String name;
    private String price;
    private byte[] image;

    public Object(String name, byte[] image, int id) {
        this.name = name;

        this.image = image;
        this.id = id;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }


    public byte[] getImage() {
        return image;
    }

    public void setImage(byte[] image) {
        this.image = image;
    }
}
