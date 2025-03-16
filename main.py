from gui import GUI

def main():
    gui = GUI()
    # gui.mount_button("Hello", lambda: print("Hello"), {'x': 10, 'y': 10})
    gui.run()
    
if __name__ == '__main__':
    main()